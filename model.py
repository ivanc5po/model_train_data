import os
import socket
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
import time

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
    return torch.tensor(tensor, dtype=torch.long).unsqueeze(0)

# 获取最大长度
max_length = max(max(len(question), len(answer)) for question, answer in zip(questions, answers))

# 构建问答模型
class QALSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, num_heads):
        super(QALSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.multihead_attn = nn.MultiheadAttention(hidden_size, num_heads)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(1, 0, 2)  # 将 batch 维度放在第二维
        x, _ = self.multihead_attn(x, x, x)  # 多头自注意力
        x, _ = self.lstm(x)
        x = self.fc(x)
        return x

def start_server(rank, world_size, device_ips, port, server_started):
    if rank == 0:
        # 启动服务器
        server_started.value = True
        print("Server started on rank 0")

def train(rank, world_size, device_ips, port, server_started):
    # 获取本机 IP 地址
    local_ip = socket.gethostbyname(socket.gethostname())
    os.environ['MASTER_ADDR'] = local_ip
    os.environ['MASTER_PORT'] = port
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    # 等待所有节点上线
    while not server_started.value:
        time.sleep(1)

    # 分布式同步点
    dist.barrier()

    # 模型参数
    input_size = len(chars)  # 输入大小为字符集大小
    hidden_size = 4096
    num_layers = 48
    output_size = len(chars)  # 输出大小与输入大小相同
    num_heads = 8  # 多头注意力的头数

    # 创建模型和优化器
    device = torch.device("cpu")
    torch.manual_seed(0)  # 为了保证可重复性，设置随机种子
    model = QALSTM(input_size, hidden_size, num_layers, output_size, num_heads).to(device)
    model = nn.parallel.DistributedDataParallel(model)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # 同步所有进程
    dist.barrier()

    # 数据集大小
    dataset_size = len(questions)

    # 训练模型
    num_epochs = 100
    print(f"start training on device {rank}......")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for i in range(rank, dataset_size, world_size):
            question_tensor = text_to_tensor(questions[i], char_to_idx, max_length).to(device)
            answer_tensor = text_to_tensor(answers[i], char_to_idx, max_length).to(device)

            optimizer.zero_grad()
            outputs = model(question_tensor)

            loss = criterion(outputs.view(-1, output_size), answer_tensor.view(-1))
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

        print(f'Device {rank} - Epoch [{epoch+1}/{num_epochs}], Average Loss: {total_loss/(dataset_size/world_size):.5f}')

if __name__ == "__main__":
    device_ips = "208.68.39.112 143.244.164.42 208.68.36.142 178.128.148.143 157.230.88.11".split()
    port = "12345"  # 设置端口号
    world_size = len(device_ips)  # 设置世界大小，即使用的设备数量
    server_started = mp.Value('b', False)
    mp.spawn(start_server, args=(world_size, device_ips, port, server_started), nprocs=1)
    mp.spawn(train, args=(world_size, device_ips, port, server_started), nprocs=world_size, join=True)
