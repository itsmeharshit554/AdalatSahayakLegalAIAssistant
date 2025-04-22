from flask import Flask, render_template, request, jsonify
from pdfminer.high_level import extract_text
from transformers import pipeline
import os

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'  # Folder to save uploaded PDFs
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Lazy loading: Initialize the summarizer as None
summarizer = None

def get_summarizer():
    """
    Lazy load the summarization pipeline.
    """
    global summarizer
    if summarizer is None:
        summarizer = pipeline('summarization', model="sshleifer/distilbart-cnn-12-6", device=-1)
    return summarizer

def clean_text(text):
    """
    Clean extracted text by removing excessive whitespace and non-printable characters.
    """
    return " ".join(text.split())

def as_summary(txt):
    """
    Summarize the given text using the Hugging Face summarization pipeline.
    """
    try:
        # Truncate the input text to 1024 characters to reduce memory usage
        truncated_text = txt[:1024]
        summarizer = get_summarizer()  # Load the model only when needed
        summary = summarizer(truncated_text, max_length=150, min_length=30, do_sample=False)
        out_text = summary[0]['summary_text']
        print(out_text)
        return out_text
    except Exception as e:
        return f"Error during summarization: {str(e)}"

def process_large_text(text, chunk_size=1024):
    """
    Split large text into smaller chunks and summarize each chunk.
    """
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    summaries = []
    for chunk in chunks:
        summary = as_summary(chunk)
        summaries.append(summary)
    return " ".join(summaries)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if text input is provided
        if 'user_text' in request.form and request.form['user_text'].strip():
            user_text = request.form['user_text']
            ai_text = as_summary(user_text)
            return jsonify({'user_text': user_text, 'ai_text': ai_text})
        
        # Check if a PDF file is uploaded
        elif 'pdf_file' in request.files:
            pdf_file = request.files['pdf_file']
            if pdf_file.filename != '':
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], pdf_file.filename)
                pdf_file.save(file_path)
                
                # Extract and clean text from the uploaded PDF
                user_text = clean_text(extract_text(file_path))
                if not user_text.strip():  # Handle case where no text is extracted
                    return jsonify({'error': 'No text could be extracted from the PDF'}), 400
                
                # Process large text in chunks
                ai_text = process_large_text(user_text)
                return jsonify({'user_text': 'Processing PDF', 'ai_text': ai_text})
        
        # If neither text nor PDF is provided
        else:
            return jsonify({'error': 'No input provided'}), 400
    
    return render_template('base.html')

if __name__ == '__main__':
    # Bind to port from environment variable for Render
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)