import json
import pickle
import numpy as np
import random
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from flask import Flask, jsonify
from flask_cors import CORS

# Loading model and pickle files
model = load_model('./models/chat_model.keras')
tokenizer = pickle.load(open('./pickle_files/tokenizer.pkl', 'rb'))
classes = pickle.load(open('./pickle_files/classes.pkl', 'rb'))
max_len = pickle.load(open('./pickle_files/max_len.pkl', 'rb'))
intents = json.load(open('./intents.json', 'r', encoding='utf-8'))

def predict_intent(sentence):
    seq = tokenizer.texts_to_sequences([sentence])
    pad = pad_sequences(seq, maxlen=max_len, padding="post")
    pred = model.predict(pad, verbose=0)[0]
    idx = np.argmax(pred)
    return classes[idx], float(pred[idx])

def get_response(intent_tag, probability):
    if probability < 0.5:
        return "Sorry, I didn't understand that."
    
    for intent in intents['intents']:
        if intent['tag'] == intent_tag:
            return random.choice(intent['responses'])
    
    return "Sorry, I don't have a response for that."

def chatbot_response(message):
    try:
        intent, prob = predict_intent(message)
        return get_response(intent, prob)
    except Exception as e:
        print(f"Chatbot error: {e}")
        return "Sorry, I'm having trouble right now."

# Flask App
app = Flask(__name__)
CORS(app)

@app.route('/')
def home():
    return jsonify({"message": "Chatbot API is running!"})

@app.route('/query/<string:sentence>')
def query_chatbot(sentence):
    decoded_message = sentence.replace("+", " ")
    response = chatbot_response(decoded_message)
    return jsonify({"query": decoded_message, "response": response})

if __name__ == '__main__':
    app.run(debug=False)