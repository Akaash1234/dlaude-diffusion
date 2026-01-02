from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
from model import Dlaude

app = Flask(__name__, static_folder="../static")
CORS(app)

model = Dlaude()
chat_history = []

@app.route('/')
def home():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def static_files(path):
    return send_from_directory(app.static_folder, path)

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    user_msg = data.get('message')
    
    if not user_msg:
        return jsonify({'error': 'no message'}), 400
        
    chat_history.append({'role': 'user', 'content': user_msg})
    
    response = model.generate(user_msg, chat_history)
    
    chat_history.append({'role': 'assistant', 'content': response})
    
    return jsonify({'response': response, 'history': chat_history})

@app.route('/api/clear', methods=['POST'])
def clear():
    global chat_history
    chat_history = []
    return jsonify({'status': 'cleared'})

if __name__ == '__main__':
    app.run(debug=True, port=8000)
