import os
import json
import logging
import traceback
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

def pad_sequence(sequence, max_length):
    if len(sequence) < max_length:
        sequence += [0] * (max_length - len(sequence))
    return sequence[:max_length]

max_length = max(len(tokenize(sentence)) for sentence in questions + answers)
np.save("max_length", max_length)

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

def train_subset(questions_subset, answers_subset, tokenizer, max_length, epoch_num, data_index):
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab_size = len(tokenizer) + 1
    hidden_size = 2048
    num_layers = 32
    num_heads = 32
    
    model = QATransformer(vocab_size, hidden_size, num_layers, num_heads).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    dataset_size = len(questions_subset)

    for epoch in range(epoch_num):
        total_loss = 0
        for i in range(dataset_size):
            question_tokens = tokenize(questions_subset[i])
            answer_tokens = tokenize(answers_subset[i])
            padded_question = pad_sequence(question_tokens, max_length)
            padded_answer = pad_sequence(answer_tokens, max_length)
            
            question_tensor = torch.tensor(padded_question).unsqueeze(0).to(device)
            answer_tensor = torch.tensor(padded_answer).unsqueeze(0).to(device)
            
            optimizer.zero_grad()
            output = model(question_tensor, answer_tensor)
            loss = nn.functional.cross_entropy(output.squeeze(0), answer_tensor.squeeze(0))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            print(f'Epoch [{epoch+1}/100], Data Index [{data_index}], Data [{i+1}/{dataset_size}], Loss: {total_loss/(i+1):.5f}')

        torch.save(model.state_dict(), 'model.pth')
        with open("tokenizer.json", "w", encoding="utf-8") as f:
            json.dump(tokenizer, f)
    
def split_data(data, n):
    chunk_size = len(data) // n
    return [data[i:i+chunk_size] for i in range(0, len(data), chunk_size)]

def train_with_data_split(questions, answers, tokenizer, max_length, n):
    question_chunks = split_data(questions, n)
    answer_chunks = split_data(answers, n)

    for i in range(n):
        train_subset(question_chunks[i], answer_chunks[i], tokenizer, max_length, 100, i+1)

if __name__ == "__main__":
    train_with_data_split(questions, answers, tokenizer, max_length, 128)
