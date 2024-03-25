import json
import torch
import numpy as np
from collections import Counter
from torch import nn
from torch.nn import functional as F

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

def tokenize(sentence, tokenizer):
    return [tokenizer[word] if word in tokenizer else 0 for word in sentence.split()]

def pad_sequence(sequence, max_length):
    if len(sequence) < max_length:
        sequence += [0] * (max_length - len(sequence))
    return sequence[:max_length]

def load_tokenizer():
    with open("tokenizer.json", "r", encoding="utf-8") as f:
        tokenizer = json.load(f)
    return tokenizer

def load_max_length():
    return int(np.load("max_length.npy"))

def load_model():
    vocab_size = len(tokenizer) + 1
    hidden_size = 2048
    num_layers = 32
    num_heads = 32

    model = QATransformer(vocab_size, hidden_size, num_layers, num_heads)
    model.load_state_dict(torch.load("model.pth"))
    model.eval()
    return model

def predict_answer(question, tokenizer, max_length, model):
    question_tokens = tokenize(question, tokenizer)
    padded_question = pad_sequence(question_tokens, max_length)

    question_tensor = torch.tensor(padded_question).unsqueeze(0)

    with torch.no_grad():
        output = model(question_tensor, question_tensor)  # Using the question as both source and target

    predicted_token_ids = torch.argmax(output, dim=2).squeeze(0).tolist()
    print(predicted_token_ids)
    predicted_answer = ' '.join([token for token in predicted_token_ids if token != 0])  # Remove padding tokens
    return predicted_answer

if __name__ == "__main__":
    tokenizer = load_tokenizer()
    max_length = load_max_length()
    model = load_model()

    while True:
        question = input("Q: ")
        predicted_answer = predict_answer(question, tokenizer, max_length, model)
        print("A:", predicted_answer)
