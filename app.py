from flask import Flask, render_template, request, jsonify
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'shadow'))
from aipha_shadow import AiphaShadow
import logging

app = Flask(__name__)

# Initialize AiphaShadow
shadow = AiphaShadow()

@app.route('/')
def index():
    """Main page with query form"""
    return render_template('index.html', available_llms=shadow.available_llms, default_llm=shadow.default_llm)

@app.route('/query', methods=['POST'])
def query():
    """Handle query submission"""
    try:
        question = request.form.get('question', '').strip()
        llm = request.form.get('llm', shadow.default_llm)

        if not question:
            return jsonify({'error': 'Please enter a question'}), 400

        # Query using AiphaShadow
        response = shadow.query(question, llm)

        return jsonify({'response': response, 'llm': llm})

    except Exception as e:
        logging.error(f"Query error: {e}")
        return jsonify({'error': 'An error occurred while processing your query'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)