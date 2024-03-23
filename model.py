import tensorflow as tf
import numpy as np
import socket
import os

# 数据集，假设有一组问题和对应的回答
questions = open("questions.txt", "r", encoding="utf-8").readlines()
answers = open("answers.txt", "r", encoding="utf-8").readlines()

# 构建字符到索引和索引到字符的映射
chars = sorted(list(set(''.join(questions + answers))))
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}

# 将文本转换为Tensor，并确保每个样本的长度一致
def text_to_tensor(text, char_to_idx, max_length):
    tensor = [char_to_idx[ch] for ch in text if ch in char_to_idx]
    tensor += [0] * (max_length - len(tensor)) 
    return np.array(tensor)

# 获取最大长度
max_length = max(max(len(question), len(answer)) for question, answer in zip(questions, answers))

# 构建问答模型
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
        return output

def train(strategy, questions, answers, char_to_idx, max_length):
    # 模型参数
    input_size = len(chars)  # 输入大小为字符集大小
    hidden_size = 128
    output_size = len(chars)  # 输出大小与输入大小相同
    num_heads = 8  # 多头注意力的头数

    # 创建模型和优化器
    with strategy.scope():
        model = QALSTM(input_size, hidden_size, output_size, num_heads)
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

    # 数据集大小
    dataset_size = len(questions)

    # 训练模型
    num_epochs = 100
    for epoch in range(num_epochs):
        total_loss = 0
        for i in range(dataset_size):
            question_tensor = text_to_tensor(questions[i], char_to_idx, max_length)
            answer_tensor = tf.reshape(answer_tensor, (-1,))

            def train_step(inputs):
                question_tensor, answer_tensor = inputs
                with tf.GradientTape() as tape:
                    output = model(question_tensor)
                    output = tf.reshape(output, (-1, output_size))
                    loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(answer_tensor, output, from_logits=True))
                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(grads, model.trainable_variables))
                return loss

            per_replica_losses = strategy.run(train_step, args=((tf.constant([question_tensor], dtype=tf.int32), tf.constant([answer_tensor], dtype=tf.int32)),))
            total_loss += strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

        print('Epoch [{}/{}], Loss: {:.5f}'.format(epoch+1, num_epochs, total_loss/dataset_size))

if __name__ == "__main__":
    # 设置分布式参数
    ip_list = ["208.68.39.112:12345", "143.244.164.42:12345", "208.68.36.142:12345", "178.128.148.143:12345", "157.230.88.11:12345"]
    num_workers = len(ip_list)
    # 设置分布式参数
    cluster_resolver = tf.distribute.cluster_resolver.TFConfigClusterResolver(ip_list)

    # 指定本地IP地址
    cluster_resolver.task_type = 'worker'

    # 指定任务ID
    strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

    # 开始训练
    train(strategy, questions, answers, char_to_idx, max_length)
