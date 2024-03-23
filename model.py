import os
import json
import logging
import traceback
import requests
import numpy as np
import tensorflow as tf

def get_public_ip():
    try:
        response = requests.get('https://api64.ipify.org/')
        if response.status_code == 200:
            return response.text
        else:
            print("error code:", response.status_code)
    except Exception as e:
        print("error:", e)
    return None

public_ip = get_public_ip()

ip_list = ['208.68.39.112:12345', '143.244.164.42:12345']
os.environ['TF_CONFIG'] = json.dumps({
    'cluster': {
        'worker': ip_list
    },
    'task': {'type': 'worker', 'index': ip_list.index(public_ip+":12345")}
})

tf.get_logger().setLevel(logging.INFO)
logger = logging.getLogger(__name__)

save_dir = 'model'

try:
    questions = open("questions.txt", "r", encoding="utf-8").readlines()
    answers = open("answers.txt", "r", encoding="utf-8").readlines()
except Exception as e:
    logger.error("Failed to read data files: %s", e)
    logger.error(traceback.format_exc())
    exit(1)

# Tokenization and Padding
tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
tokenizer.fit_on_texts(questions + answers)
questions_sequences = tokenizer.texts_to_sequences(questions)
answers_sequences = tokenizer.texts_to_sequences(answers)
max_length = max(max(len(seq) for seq in questions_sequences), max(len(seq) for seq in answers_sequences))
questions_padded = tf.keras.preprocessing.sequence.pad_sequences(questions_sequences, maxlen=max_length, padding='post')
answers_padded = tf.keras.preprocessing.sequence.pad_sequences(answers_sequences, maxlen=max_length, padding='post')

class QALSTM(tf.keras.Model):
    def __init__(self, vocab_size, hidden_size, output_size, num_heads):
        super(QALSTM, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = tf.keras.layers.Embedding(vocab_size, hidden_size)
        self.multihead_attn = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=hidden_size, value_dim=hidden_size)  
        self.lstm = tf.keras.layers.LSTM(hidden_size, return_sequences=True, dtype=tf.float32)  
        self.fc = tf.keras.layers.Dense(output_size)

    def call(self, x):
        x = self.embedding(x)
        attn_output = self.multihead_attn(x, x, x)
        lstm_output = self.lstm(attn_output)
        output = self.fc(lstm_output)
        
        output = tf.squeeze(output, axis=0)
        return output

def train(strategy, questions, answers, tokenizer, max_length):
    vocab_size = len(tokenizer.word_index) + 1
    hidden_size = 128
    output_size = vocab_size
    num_heads = 8

    with strategy.scope():
        model = QALSTM(vocab_size, hidden_size, output_size, num_heads)
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

    dataset_size = len(questions)

    @tf.function
    def train_step(question_tensor, answer_tensor):
        with tf.GradientTape() as tape:
            output = model(question_tensor)
            output = tf.expand_dims(output, axis=0)
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
            question_tensor = questions[i]
            answer_tensor = answers[i]
            try:
                loss = strategy.run(train_step, args=(tf.constant([question_tensor]), tf.constant([answer_tensor])))
                total_loss += loss
                print('Epoch [{}/{}], data [{}/{}], Loss: {:.5f}'.format(epoch+1, num_epochs, i+1, dataset_size, total_loss/(i+1)))
            except Exception as e:
                logger.error("Error in training step: %s", e)
                logger.error(traceback.format_exc())

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
            train(strategy, questions_padded, answers_padded, tokenizer, max_length)
    except Exception as e:
        logger.error("Error during training: %s", e)
        logger.error(traceback.format_exc())
