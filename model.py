import torch
import torch.nn as nn
import torch.optim as optim

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
    tensor += [0] * (max_length - len(tensor))  # 使用0进行padding
    return torch.tensor(tensor, dtype=torch.long).unsqueeze(0)

# 获取最大长度
max_length = max(max(len(question), len(answer)) for question, answer in zip(questions, answers))

# 构建问答模型
class QALSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(QALSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = self.fc(x)
        return x

# 模型参数
input_size = len(chars)  # 输入大小为字符集大小
hidden_size = 2048
num_layers = 24
output_size = len(chars)  # 输出大小与输入大小相同

# 创建模型和优化器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = QALSTM(input_size, hidden_size, num_layers, output_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 数据集大小
dataset_size = len(questions)

# 训练模型
num_epochs = 100

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for i in range(dataset_size):
        question_tensor = text_to_tensor(questions[i], char_to_idx, max_length).to(device)
        answer_tensor = text_to_tensor(answers[i], char_to_idx, max_length).to(device)

        optimizer.zero_grad()
        outputs = model(question_tensor)

        loss = criterion(outputs.view(-1, output_size), answer_tensor.view(-1))
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    avg_loss = total_loss / dataset_size
    print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}')

def generate_answer(model, question, max_length, output_size):
    model.eval()
    with torch.no_grad():
        question_tensor = text_to_tensor(question, char_to_idx, max_length).to(device)
        output_text = question

        outputs = model(question_tensor)

        for _ in range(output_size):
            output_dist = outputs.squeeze(0).div(0.8).exp()
            top_i = torch.multinomial(output_dist, 1)[0]
            predicted_char = idx_to_char[top_i.item()]
            output_text += predicted_char

            question_tensor = torch.tensor([[top_i]], dtype=torch.long).to(device)
            outputs = model(question_tensor)

        return output_text

while True:
    question = input("Q: ")
    output_size = 200
    generated_answer = generate_answer(model, question, max_length, output_size)
    print("A:", generated_answer.replace(r"[\n]" ,"\n"))
