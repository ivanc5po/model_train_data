import tensorflow as tf
import numpy as np

# Data Preparation
questions = open("questions.txt", "r", encoding="utf-8").readlines()
answers = open("answers.txt", "r", encoding="utf-8").readlines()

chars = sorted(list(set(''.join(questions + answers))))
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}

def text_to_tensor(text, char_to_idx, max_length):
    tensor = [char_to_idx[ch] for ch in text if ch in char_to_idx]
    tensor += [0] * (max_length - len(tensor)) 
    return np.array(tensor)

max_length = max(max(len(question), len(answer)) for question, answer in zip(questions, answers))

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
        with tf.GradientTape() as tape:
            output = model(question_tensor)
            output = tf.expand_dims(output, axis=0)  # Adding back batch dimension
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
            loss = strategy.run(train_step, args=(tf.constant([question_tensor], dtype=tf.int32), tf.constant([answer_tensor], dtype=tf.int32)))
            total_loss += loss

        print('Epoch [{}/{}], Loss: {:.5f}'.format(epoch+1, num_epochs, total_loss/dataset_size))

if __name__ == "__main__":
    ip_list = ["208.68.39.112:12345", "143.244.164.42:12345", "208.68.36.142:12345", "178.128.148.143:12345", "157.230.88.11:12345"]
    cluster_resolver = tf.distribute.cluster_resolver.TFConfigClusterResolver(ip_list)
    cluster_resolver.task_type = 'worker'
    strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy(
        communication=tf.distribute.experimental.CollectiveCommunication.AUTO)

    train(strategy, questions, answers, char_to_idx, max_length)
