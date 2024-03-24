import os
import json
import logging
import traceback
import requests
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter
from multiprocessing import Process

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
os.environ['MASTER_ADDR'] = ip_list[0].split(":")[0]
os.environ['MASTER_PORT'] = ip_list[0].split(":")[1]
os.environ['WORLD_SIZE'] = str(len(ip_list))
os.environ['RANK'] = str(ip_list.index(public_ip+":12345"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

save_dir = 'model'

with open("questions.txt", "r", encoding="utf-8") as f:
    questions = f.readlines()
with open("answers.txt", "r", encoding="utf-8") as f:
    answers = f.readlines()

# Tokenization and Padding
word_counts = Counter()
for sentence in questions + answers:
    word_counts.update(sentence.split())
tokenizer = {word: index + 1 for index, (word, _) in enumerate(word_counts.most_common())}

def tokenize(sentence):
    return [tokenizer[word] for word in sentence.split()]

max_length = max(len(tokenize(sentence)) for sentence in questions + answers)
tokenizer = torch.nn.utils.rnn.pad_sequence([torch.tensor(tokenize(sentence)) for sentence in questions + answers], batch_first=True)

class QATransformer(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers, num_heads, dropout=0.1):
        super(QATransformer, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.transformer = nn.Transformer(d_model=hidden_size, nhead=num_heads, num_encoder_layers=num_layers, num_decoder_layers=num_layers, dropout=dropout)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, src, tgt):
        src = self.embedding(src)
        tgt = self.embedding(tgt)
        output = self.transformer(src, tgt)
        output = self.fc(output)
        return output

def train(rank, world_size, questions, answers, tokenizer, max_length):
    print("starting......")
    torch.manual_seed(0)
    device = torch.device("cpu")

    print("1")
    vocab_size = len(tokenizer) + 1
    hidden_size = 128
    num_layers = 2
    num_heads = 2

    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    
    print("2")
    model = QATransformer(vocab_size, hidden_size, num_layers, num_heads).to(device)
    model = DDP(model)

    print("3")
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    dataset_size = len(questions)

    print("4")
    for epoch in range(100):
        total_loss = 0
        for i in range(dataset_size):
            question_tensor = questions[i].unsqueeze(0).to(device)
            answer_tensor = answers[i].unsqueeze(0).to(device)
            optimizer.zero_grad()
            output = model(question_tensor, answer_tensor)
            loss = nn.functional.cross_entropy(output.squeeze(0), answer_tensor.squeeze(0))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            print(f'Rank [{rank+1}/{world_size}], Epoch [{epoch+1}/100], Data [{i+1}/{dataset_size}], Loss: {total_loss/(i+1):.5f}')

        if rank == 0:
            if not os.path.exists(save_dir):
                try:
                    os.makedirs(save_dir)
                except Exception as e:
                    logger.error("Failed to create directory: %s", e)
                    logger.error(traceback.format_exc())

            try:
                torch.save(model.module.state_dict(), os.path.join(save_dir, 'qatransformer_model.pth'))
            except Exception as e:
                logger.error("Failed to save model: %s", e)
                logger.error(traceback.format_exc())

if __name__ == "__main__":
    world_size = int(os.environ['WORLD_SIZE'])
    processes = []
    for rank in range(world_size):
        p = Process(target=train, args=(rank, world_size, questions, answers, tokenizer, max_length))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
        
