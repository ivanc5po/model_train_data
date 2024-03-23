import os
import json
import logging
import traceback
import requests
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader

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

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

public_ip = get_public_ip()

ip_list = ['208.68.39.112:12345', '143.244.164.42:12345']
os.environ['RANK'] = str(ip_list.index(public_ip+":12345"))
os.environ['WORLD_SIZE'] = str(len(ip_list))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

save_dir = 'model'

try:
    with open("questions.txt", "r", encoding="utf-8") as f:
        questions = f.readlines()
    with open("answers.txt", "r", encoding="utf-8") as f:
        answers = f.readlines()
except Exception as e:
    logger.error("Failed to read data files: %s", e)
    logger.error(traceback.format_exc())
    exit(1)

# Tokenization and Padding
tokenizer = torch.jit.load('tokenizer.pth')
questions_sequences = tokenizer(questions)
answers_sequences = tokenizer(answers)
max_length = max(max(len(seq) for seq in questions_sequences), max(len(seq) for seq in answers_sequences))
questions_padded = torch.nn.utils.rnn.pad_sequence(questions_sequences, batch_first=True, padding_value=0)
answers_padded = torch.nn.utils.rnn.pad_sequence(answers_sequences, batch_first=True, padding_value=0)

class QALSTM(nn.Module):
    def __init__(self, vocab_size, hidden_size, output_size, num_heads):
        super(QALSTM, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.multihead_attn = nn.MultiheadAttention(hidden_size, num_heads=num_heads)  
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)  
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(1, 0, 2)  # Change from (batch_size, seq_len, hidden_size) to (seq_len, batch_size, hidden_size) for LSTM
        attn_output, _ = self.multihead_attn(x, x, x)
        attn_output = attn_output.permute(1, 0, 2)  # Change back to (batch_size, seq_len, hidden_size)
        lstm_output, _ = self.lstm(attn_output)
        output = self.fc(lstm_output)
        return output

def train(model, device, train_loader, optimizer, criterion):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        output = output.permute(0, 2, 1)  # Change from (batch_size, seq_len, vocab_size) to (batch_size, vocab_size, seq_len) for loss calculation
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

def main(rank, world_size):
    try:
        setup(rank, world_size)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        vocab_size = tokenizer.vocab_size
        hidden_size = 128
        output_size = vocab_size
        num_heads = 8

        model = QALSTM(vocab_size, hidden_size, output_size, num_heads).to(device)
        model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])

        optimizer = optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()

        dataset = YourDataset(questions_padded, answers_padded)  # Define your dataset class and initialization
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        train_loader = DataLoader(dataset, batch_size=64, sampler=train_sampler)

        num_epochs = 100
        for epoch in range(num_epochs):
            train(model, device, train_loader, optimizer, criterion)
            print('Epoch [{}/{}]'.format(epoch+1, num_epochs))

        if not os.path.exists(save_dir):
            try:
                os.makedirs(save_dir)
            except Exception as e:
                logger.error("Failed to create directory: %s", e)
                logger.error(traceback.format_exc())

        try:
            torch.save(model.state_dict(), os.path.join(save_dir, 'qalstm_model.pth'))
        except Exception as e:
            logger.error("Failed to save model: %s", e)
            logger.error(traceback.format_exc())

    except Exception as e:
        logger.error("Error during training: %s", e)
        logger.error(traceback.format_exc())

    finally:
        cleanup()

if __name__ == "__main__":
    world_size = int(os.environ['WORLD_SIZE'])
    mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)
