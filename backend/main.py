import json
import pickle
import numpy as np
import random
from keras.models import load_model
from flask import Flask, jsonify
from flask_cors import CORS

import nltk
from nltk.stem import WordNetLemmatizer

# Download NLTK data once
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

lemmatizer = WordNetLemmatizer()

# Loading model and data
model = load_model('./models/chatbot_model.keras')
intents = json.load(open('./intents.json', 'r', encoding='utf-8'))
words = pickle.load(open('./pickle_files/words.pkl', 'rb'))
classes = pickle.load(open('./pickle_files/classes.pkl', 'rb'))

def clean_up_sentence(sentence):
    """Tokenize and lemmatize sentence"""
    tokens = nltk.word_tokenize(sentence)
    return [lemmatizer.lemmatize(token.lower()) for token in tokens]

def create_bow(sentence, words):
    """Create bag-of-words vector"""
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    
    for i, word in enumerate(words):
        if word in sentence_words:
            bag[i] = 1
    return np.array(bag)

def predict_intent(sentence, model, threshold=0.25):
    """Predict intent with probability threshold"""
    bow_vector = create_bow(sentence, words)
    prediction = model.predict(np.array([bow_vector]), verbose=0)[0]
    
    # Filter results above threshold
    results = [
        {"intent": classes[i], "probability": float(prob)}
        for i, prob in enumerate(prediction) 
        if prob > threshold
    ]
    # Sort by probability (highest first)
    return sorted(results, key=lambda x: x["probability"], reverse=True)

def get_response(intents_list, intents_data):
    """Get random response for highest probability intent"""
    if not intents_list:
        return "Sorry, I didn't understand that."
    
    best_intent = intents_list[0]['intent']
    
    for intent in intents_data['intents']:
        if intent['tag'] == best_intent:
            return random.choice(intent['responses'])
    
    return "Sorry, I don't have a response for that."

def chatbot_response(message):
    """Main chatbot processing function"""
    try:
        predicted_intents = predict_intent(message, model)
        return get_response(predicted_intents, intents)
    except Exception as e:
        print(f"Chatbot error: {e}")
        return "Sorry, I'm having trouble right now."

# Flask App
app = Flask(__name__)
# CORS(app)
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route('/')
def home():
    return jsonify({"message": "Chatbot API is running!"})

@app.route('/query/<string:sentence>')
def query_chatbot(sentence):
    """Main API endpoint"""
    # Decode URL spaces (replace + with space)
    decoded_message = sentence.replace("+", " ")
    response = chatbot_response(decoded_message)
    
    return jsonify({
        "query": decoded_message,
        "response": response
    })

if __name__ == '__main__':
    app.run(debug=False)