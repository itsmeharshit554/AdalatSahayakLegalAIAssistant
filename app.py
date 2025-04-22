from flask import Flask, render_template, request, jsonify
from pdfminer.high_level import extract_text
from transformers import pipeline
import os

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

summarizer = None  # Lazy-loaded summarizer

def load_summarizer():
    global summarizer
    if summarizer is None:
        summarizer = pipeline(
            'summarization',
            model="sshleifer/distilbart-cnn-12-6",
            device=-1  # CPU only
        )

@app.route('/', methods=['GET', 'POST'])
def index():
    user_text = None
    ai_text = None

    if request.method == 'POST':
        # Text input
        if 'user_text' in request.form and request.form['user_text'].strip():
            user_text = request.form['user_text'][:2000]  # Limit input size
            ai_text = as_summary(user_text)
            return jsonify({'user_text': user_text, 'ai_text': ai_text})
        
        # PDF input
        elif 'pdf_file' in request.files:
            pdf_file = request.files['pdf_file']
            if pdf_file and pdf_file.filename:
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], pdf_file.filename)
                pdf_file.save(file_path)
                
                user_text = extract_text(file_path)
                if not user_text.strip():
                    return jsonify({'error': 'No text could be extracted from the PDF'}), 400
                user_text = user_text[:2000]  # Limit extracted text length
                ai_text = as_summary(user_text)
                return jsonify({'user_text': 'Processing PDF', 'ai_text': ai_text})
        
        return jsonify({'error': 'No input provided'}), 400

    return render_template('base.html')

def as_summary(txt):
    try:
        load_summarizer()
        summary = summarizer(txt, max_length=60, min_length=20, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        return f"Error during summarization: {str(e)}"

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
