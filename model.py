import tensorflow as tf
import numpy as np
import socket

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
    def __init__(self, input_size, hidden_size, num_layers, output_size, num_heads):
        super(QALSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = tf.keras.layers.Embedding(input_size, hidden_size)
        self.multihead_attn = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=hidden_size)
        self.lstm = tf.keras.layers.LSTM(hidden_size, return_sequences=True, num_layers=num_layers)  # Changed num_layers here
        self.fc = tf.keras.layers.Dense(output_size)

    def call(self, x):
        x = self.embedding(x)
        attn_output = self.multihead_attn(x, x, x)
        lstm_output = self.lstm(attn_output)
        output = self.fc(lstm_output)
        return output

def train(rank, world_size, device_ips, port):
    # 获取本机 IP 地址
    local_ip = socket.gethostbyname(socket.gethostname())

    # 设置分布式参数
    cluster_spec = tf.train.ClusterSpec({"worker": device_ips})
    server = tf.distribute.Server(cluster_spec, job_name="worker", task_index=rank)

    # 连接到集群
    tf.config.experimental_connect_to_cluster(server.target)
    tf.distribute.experimental_set_strategy(tf.distribute.experimental.MultiWorkerMirroredStrategy())

    print("正在載入模型, 節點:", rank)
    
    # 模型参数
    input_size = len(chars)  # 输入大小为字符集大小
    hidden_size = 128  # Changed hidden_size here
    num_layers = 8  # Changed num_layers here
    output_size = len(chars)  # 输出大小与输入大小相同
    num_heads = 8  # 多头注意力的头数

    # 创建模型和优化器
    with tf.device("/job:worker/task:{}".format(rank)):
        model = QALSTM(input_size, hidden_size, num_layers, output_size, num_heads)
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

    # 数据集大小
    dataset_size = len(questions)

    # 训练模型
    num_epochs = 100
    print("開始訓練, 節點:", rank)
    for epoch in range(num_epochs):
        total_loss = 0
        for i in range(rank, dataset_size, world_size):
            question_tensor = text_to_tensor(questions[i], char_to_idx, max_length)
            answer_tensor = text_to_tensor(answers[i], char_to_idx, max_length)

            with tf.GradientTape() as tape:
                output = model(tf.constant([question_tensor], dtype=tf.int32))
                loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(answer_tensor, output, from_logits=True))

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            total_loss += loss.numpy()

        print('Device {} - Epoch [{}/{}], Loss: {:.5f}'.format(rank, epoch+1, num_epochs, total_loss/dataset_size))

if __name__ == "__main__":
    device_ips = ["208.68.39.112", "143.244.164.42", "208.68.36.142", "178.128.148.143", "157.230.88.11"]
    port = "12345"  # 设置端口号
    world_size = len(device_ips)  # 设置世界大小，即使用的设备数量
    for i in range(world_size):
        train(i, world_size, device_ips, port)
