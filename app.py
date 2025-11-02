from flask import Flask, render_template, request, jsonify, session
import sys
import os
import json
from datetime import datetime
sys.path.append(os.path.join(os.path.dirname(__file__), 'shadow'))
from aipha_shadow_simple import AiphaShadowSimple
import logging

app = Flask(__name__)
app.secret_key = 'aipha_shadow_secret_key_2025'  # Clave para sesiones

# Global conversation memory storage
conversation_history = {}

# Initialize AiphaShadowSimple
shadow = AiphaShadowSimple()

@app.route('/')
def index():
    """Main page with query form"""
    return render_template('index.html', available_llms=shadow.available_llms, default_llm=shadow.default_llm)

@app.route('/clear_conversation', methods=['POST'])
def clear_conversation():
    """Clear conversation history for current session"""
    session_id = session.get('session_id')
    if session_id and session_id in conversation_history:
        del conversation_history[session_id]
    return jsonify({'status': 'conversation_cleared'})

@app.route('/get_conversation_history', methods=['GET'])
def get_conversation_history():
    """Get conversation history for current session"""
    session_id = session.get('session_id')
    if session_id and session_id in conversation_history:
        return jsonify({
            'status': 'success',
            'history': conversation_history[session_id]
        })
    return jsonify({'status': 'success', 'history': []})

@app.route('/query', methods=['POST'])
def query():
    """Handle query submission with conversation memory"""
    try:
        question = request.form.get('question', '').strip()
        llm = request.form.get('llm', shadow.default_llm)

        if not question:
            return jsonify({'error': 'Please enter a question'}), 400

        # Get or create session ID
        session_id = session.get('session_id')
        if not session_id:
            session_id = f"session_{datetime.now().timestamp()}"
            session['session_id'] = session_id

        # Initialize conversation history for this session if it doesn't exist
        if session_id not in conversation_history:
            conversation_history[session_id] = []

        # Add user question to conversation history
        conversation_history[session_id].append({
            'role': 'user',
            'content': question,
            'timestamp': datetime.now().isoformat()
        })

        # Get conversation history to include context
        conversation_context = conversation_history[session_id][-10:]  # Last 10 exchanges

        # Prepare conversation history for LLM (format as conversation summary)
        conversation_summary = ""
        if len(conversation_context) > 1:
            conversation_summary = "HISTORIAL DE CONVERSACIÓN PREVIA:\n"
            for i, msg in enumerate(conversation_context[:-1]):  # Exclude current question
                role = "Usuario" if msg['role'] == 'user' else "Shadow_1.0"
                conversation_summary += f"{role}: {msg['content']}\n"
            conversation_summary += "\n"

        # Combine current question with conversation history
        enhanced_question = f"{conversation_summary}PREGUNTA ACTUAL: {question}"

        # Query using AiphaShadowSimple with conversation memory
        response = shadow.query_with_memory(enhanced_question, llm, conversation_context)

        # Add LLM response to conversation history
        conversation_history[session_id].append({
            'role': 'assistant',
            'content': response,
            'timestamp': datetime.now().isoformat(),
            'llm_used': llm
        })

        # Clean up old conversation history (keep only last 20 exchanges)
        if len(conversation_history[session_id]) > 20:
            conversation_history[session_id] = conversation_history[session_id][-20:]

        return jsonify({
            'response': response,
            'llm': llm,
            'conversation_length': len(conversation_history[session_id])
        })

    except Exception as e:
        logging.error(f"Query error: {e}")
        return jsonify({'error': f'An error occurred while processing your query: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)