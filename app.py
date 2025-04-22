from flask import Flask, render_template, request, jsonify
from pdfminer.high_level import extract_text
from transformers import pipeline
import os

# Optional: For extractive fallback
from summa import summarizer as summa_summarizer

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 * 1024  # 5MB max upload

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load summarizer globally
print("Loading summarizer model...")
summarizer = pipeline(
    'summarization',
    model="sshleifer/distilbart-cnn-12-6",
    device=-1  # Use CPU
)
print("Summarizer ready.")

# Helper: Chunk long text
def chunk_text(text, max_len=1000):
    sentences = text.split('. ')
    chunks, chunk = [], ''
    for sentence in sentences:
        if len(chunk) + len(sentence) < max_len:
            chunk += sentence + '. '
        else:
            chunks.append(chunk.strip())
            chunk = sentence + '. '
    if chunk:
        chunks.append(chunk.strip())
    return chunks

# Helper: Run summarization
def as_summary(txt):
    try:
        if not txt.strip():
            return "No text provided."

        chunks = chunk_text(txt)
        summaries = summarizer(
            chunks,
            max_length=60,
            min_length=20,
            do_sample=False
        )

        return ' '.join([s['summary_text'] for s in summaries])
    except Exception as e:
        print(f"Error with transformer summarization: {e}")
        # Optional fallback to extractive
        try:
            return summa_summarizer.summarize(txt, ratio=0.2)
        except Exception as fallback_error:
            return f"Error during summarization: {str(fallback_error)}"

# Route for form + summary
@app.route('/', methods=['GET', 'POST'])
def index():
    user_text = None
    ai_text = None

    if request.method == 'POST':
        # Text input
        if 'user_text' in request.form and request.form['user_text'].strip():
            user_text = request.form['user_text'][:2000]
            ai_text = as_summary(user_text)
            return jsonify({'user_text': user_text, 'ai_text': ai_text})

        # PDF upload
        elif 'pdf_file' in request.files:
            pdf_file = request.files['pdf_file']
            if pdf_file and pdf_file.filename:
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], pdf_file.filename)
                pdf_file.save(file_path)

                user_text = extract_text(file_path)
                if not user_text.strip():
                    return jsonify({'error': 'No text could be extracted from the PDF'}), 400
                user_text = user_text[:2000]
                ai_text = as_summary(user_text)
                return jsonify({'user_text': 'Processing PDF', 'ai_text': ai_text})

        return jsonify({'error': 'No input provided'}), 400

    return render_template('base.html')  # Ensure base.html exists

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
