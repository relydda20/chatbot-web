from flask import render_template, request, jsonify
from app import app
from chatbot import classify_intent, chatbot_loop
import random

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    user_input = request.form['user_input']
    intent = classify_intent(user_input)
    response = chatbot_loop(user_input, intent)
    return jsonify({'bot_response': response})