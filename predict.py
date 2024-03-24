import torch
import torch.nn as nn
import json

# Define the QATransformer model class
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
    return [tokenizer[word] for word in sentence.split()]

def pad_sequence(sequence, max_length):
    if len(sequence) < max_length:
        sequence += [0] * (max_length - len(sequence))
    return sequence[:max_length]

def load_model(model_path, vocab_size, hidden_size, num_layers, num_heads):
    model = QATransformer(vocab_size, hidden_size, num_layers, num_heads)
    model.load_state_dict(torch.load(model_path))
    return model

def predict(model, question, tokenizer, max_length):
    question_tokens = tokenize(question, tokenizer)
    padded_question = pad_sequence(question_tokens, max_length)
    question_tensor = torch.tensor(padded_question).unsqueeze(0)  # Add batch dimension
    output = model(question_tensor, question_tensor)  # Use question as both src and tgt
    predicted_index = torch.argmax(output, dim=-1).squeeze(0)  # Remove batch dimension and get predicted index
    predicted_words = [word for word, index in tokenizer.items() if index == predicted_index.item()]
    return ' '.join(predicted_words)

if __name__ == "__main__":
    # Load tokenizer
    with open("tokenizer.json", "r", encoding="utf-8") as f:
        tokenizer = json.load(f)
    
    # Load model
    model_path = 'model/qatransformer_model.pth'
    vocab_size = len(tokenizer) + 1  # Add 1 for padding token
    hidden_size = 1024
    num_layers = 24
    num_heads = 8
    max_length = 100  # Set maximum length for padding

    model = load_model(model_path, vocab_size, hidden_size, num_layers, num_heads)
    model.eval()

    # Continuous prediction loop
    while True:
        question = input("Enter your question (or 'exit' to quit): ")
        if question.lower() == 'exit':
            print("Exiting...")
            break
        prediction = predict(model, question, tokenizer, max_length)
        print("Predicted Answer:", prediction)
