import tensorflow as tf
import numpy as np

# Load the model
model = tf.keras.models.load_model('saved_model/qalstm_model')

# Data Preparation
questions = open("questions.txt", "r", encoding="utf-8").readlines()
answers = open("answers.txt", "r", encoding="utf-8").readlines()

chars = sorted(list(set(''.join(questions + answers))))
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}

def text_to_tensor(text, char_to_idx, max_length):
    tensor = [char_to_idx[ch] for ch in text if ch in char_to_idx]
    tensor += [0] * (max_length - len(tensor)) 
    return np.array(tensor)

max_length = max(max(len(question), len(answer)) for question, answer in zip(questions, answers))

# Continuous prediction
while True:
    user_input = input("Enter a question (type 'exit' to quit): ")
    if user_input.lower() == 'exit':
        break
    
    question_tensor = text_to_tensor(user_input, char_to_idx, max_length)
    question_tensor = np.expand_dims(question_tensor, axis=0)  # Add batch dimension
    prediction = model.predict(question_tensor)
    predicted_answer = ''.join([idx_to_char[idx] for idx in np.argmax(prediction[0], axis=1)])
    print("Predicted answer:", predicted_answer)
