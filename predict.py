import torch
import torch.nn as nn
import numpy as np
from collections import Counter

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

def tokenize(sentence):
    return [tokenizer[word] for word in sentence.split()]

def pad_sequence(sequence, max_length):
    if len(sequence) < max_length:
        sequence += [0] * (max_length - len(sequence))
    return sequence[:max_length]

def load_model(model_path, vocab_size, hidden_size, num_layers, num_heads):
    model = QATransformer(vocab_size, hidden_size, num_layers, num_heads)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def predict(question, model, tokenizer, max_length):
    question_tokens = tokenize(question)
    padded_question = pad_sequence(question_tokens, max_length)
    question_tensor = torch.tensor(padded_question).unsqueeze(0)
    output = model(question_tensor, question_tensor)  # Using the question itself as the target
    predicted_answer_indices = output.squeeze(0).argmax(dim=-1)
    predicted_answer = ' '.join([word for index, word in enumerate(predicted_answer_indices) if word != 0])
    return predicted_answer

if __name__ == "__main__":
    # Define model parameters
    vocab_size = np.load("max_length.npy")
    hidden_size = 2048
    num_layers = 32
    num_heads = 32
    model_path = 'models/model.pth'  # Path to the trained model

    # Load the trained model
    model = load_model(model_path, vocab_size, hidden_size, num_layers, num_heads)

    # Input loop
    while True:
        # Get question from user
        question = input("Enter your question (type 'exit' to quit): ")

        # Check if user wants to exit
        if question.lower() == 'exit':
            print("Exiting...")
            break

        # Make prediction
        predicted_answer = predict(question, model, tokenizer, max_length)
        print("Predicted answer:", predicted_answer)
