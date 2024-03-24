import torch
from model_script import QATransformer, tokenize, pad_sequence

def load_model_and_predict(question, model_path, max_length):
    # Define model parameters
    vocab_size = 10000  # Assuming you know the vocab size beforehand
    hidden_size = 2048
    num_layers = 32
    num_heads = 32

    # Load the trained model
    model = QATransformer(vocab_size, hidden_size, num_layers, num_heads)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Tokenize and pad the question
    question_tokens = tokenize(question)
    padded_question = pad_sequence(question_tokens, max_length)
    question_tensor = torch.tensor(padded_question).unsqueeze(0)

    # Predict answer
    with torch.no_grad():
        output = model(question_tensor, question_tensor)  # Using the question itself as the target
        predicted_answer_indices = output.squeeze(0).argmax(dim=-1)
        predicted_answer = ' '.join([str(index.item()) for index in predicted_answer_indices if index.item() != 0])

    return predicted_answer

if __name__ == "__main__":
    model_path = 'models/model.pth'  # Path to the trained model
    max_length = 50  # Assuming max_length is known

    # Input loop
    while True:
        # Get question from user
        question = input("Enter your question (type 'exit' to quit): ")

        # Check if user wants to exit
        if question.lower() == 'exit':
            print("Exiting...")
            break

        # Make prediction
        predicted_answer = load_model_and_predict(question, model_path, max_length)
        print("Predicted answer:", predicted_answer)
