import os
import json
import logging
import traceback
import requests
import numpy as np
import tensorflow as tf

def get_public_ip():
    try:
        # 使用 ipify 的 API 查询公共 IP 地址
        response = requests.get('https://api64.ipify.org/')
        if response.status_code == 200:
            return response.text
        else:
            print("error code：", response.status_code)
    except Exception as e:
        print("error：", e)
    return None

public_ip = get_public_ip()

ip_list = ['208.68.39.112:12345', '143.244.164.42:12345']
os.environ['TF_CONFIG'] = json.dumps({
    'cluster': {
        'worker': ip_list
    },
    'task': {'type': 'worker', 'index': ip_list.index(public_ip+":12345")}
})

# Set TensorFlow logging level
tf.get_logger().setLevel(logging.INFO)

# Define a logger
logger = logging.getLogger(__name__)

save_dir = 'model'

# Data Preparation
try:
    questions = open("questions.txt", "r", encoding="utf-8").readlines()
    answers = open("answers.txt", "r", encoding="utf-8").readlines()
except Exception as e:
    logger.error("Failed to read data files: %s", e)
    logger.error(traceback.format_exc())
    exit(1)

chars = sorted(list(set(''.join(questions + answers))))
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}

def text_to_tensor(text, char_to_idx, max_length):
    tensor = [char_to_idx[ch] for ch in text if ch in char_to_idx]
    tensor += [0] * (max_length - len(tensor))
    return np.array(tensor)

max_length = max(max(len(question), len(answer)) for question, answer in zip(questions, answers))

def cast_to_int32(tensor):
    return tf.cast(tensor, tf.int32)
    
# Model Definition
class QALSTM(tf.keras.Model):
    def __init__(self, input_size, hidden_size, output_size, num_heads):
        super(QALSTM, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = tf.keras.layers.Embedding(input_size, hidden_size)
        self.multihead_attn = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=hidden_size)
        self.lstm = tf.keras.layers.LSTM(hidden_size, return_sequences=True)
        self.fc = tf.keras.layers.Dense(output_size)

    def call(self, x):
        x = self.embedding(x)
        attn_output = self.multihead_attn(x, x, x)
        lstm_output = self.lstm(attn_output)
        output = self.fc(lstm_output)
        
        output = tf.squeeze(output, axis=0)
        return output

# Training Function
def train(strategy, questions, answers, char_to_idx, max_length):
    input_size = len(chars)
    hidden_size = 128
    output_size = len(chars)
    num_heads = 8

    with strategy.scope():
        model = QALSTM(input_size, hidden_size, output_size, num_heads)
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

    dataset_size = len(questions)

    @tf.function
    def train_step(question_tensor, answer_tensor):
        question_tensor = tf.cast(question_tensor, tf.float32)  # 将张量转换为 float32 类型
        answer_tensor = tf.cast(answer_tensor, tf.float32)  # 将张量转换为 float32 类型
    
        with tf.GradientTape() as tape:
            output = model(question_tensor)
            output = tf.expand_dims(output, axis=0)  # 添加回批量维度
            expected_shape = tf.shape(answer_tensor)
            output_shape = tf.shape(output)
            pad_size = tf.maximum(expected_shape[1] - output_shape[1], 0)
            paddings = [[0, 0], [0, pad_size], [0, 0]]
            output = tf.pad(output, paddings, constant_values=0.0)
            output = output[:, :expected_shape[1], :]
            loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(answer_tensor, output, from_logits=True))
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss
        
    num_epochs = 100
    for epoch in range(num_epochs):
        total_loss = 0
        for i in range(dataset_size):
            question_tensor = text_to_tensor(questions[i], char_to_idx, max_length)
            answer_tensor = text_to_tensor(answers[i], char_to_idx, max_length)
            try:
                loss = strategy.run(train_step, args=(tf.constant([question_tensor], dtype=tf.int32), tf.constant([answer_tensor], dtype=tf.int32)))
                total_loss += loss
                print('Epoch [{}/{}], data [{}/{}], Loss: {:.5f}'.format(epoch+1, num_epochs, i+1, dataset_size, total_loss/(i+1)))
            except Exception as e:
                logger.error("Error occurred during training step: %s", e)
                logger.error(traceback.format_exc())
                # You can choose to continue training or exit the program based on the severity of the error

        if not os.path.exists(save_dir):
            try:
                os.makedirs(save_dir)
            except Exception as e:
                logger.error("Failed to create directory: %s", e)
                logger.error(traceback.format_exc())

        try:
            model.save(os.path.join(save_dir, 'qalstm_model'))
        except Exception as e:
            logger.error("Failed to save model: %s", e)
            logger.error(traceback.format_exc())

if __name__ == "__main__":
    try:
        strategy = tf.distribute.MultiWorkerMirroredStrategy()
        with strategy.scope():
            train(strategy, questions, answers, char_to_idx, max_length)
    except Exception as e:
        logger.error("Error occurred during training: %s", e)
        logger.error(traceback.format_exc())
