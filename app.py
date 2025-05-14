from flask import Flask, render_template, request, jsonify
from transformers import pipeline

app = Flask(__name__)
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    try:
        data = request.get_json()
        input_text = data.get('text', '')
        if not input_text.strip():
            return jsonify({'summary': ''})
        result = summarizer(input_text, max_length=130, min_length=30, do_sample=False)
        summary = result[0]['summary_text']
        return jsonify({'summary': summary})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
