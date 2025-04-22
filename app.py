from flask import Flask, render_template, request, jsonify
from pdfminer.high_level import extract_text
from transformers import pipeline
import os

# Initialize the summarization pipeline with a lightweight model
summarizer = pipeline('summarization', model="sshleifer/distilbart-cnn-12-6", revision="main")

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'  # Folder to save uploaded PDFs
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
@app.route('/', methods=['GET', 'POST'])
def index():
    user_text = None
    ai_text = None

    if request.method == 'POST':
        # Check if text input is provided
        if 'user_text' in request.form and request.form['user_text'].strip():
            user_text = request.form['user_text']
            ai_text = as_summary(user_text)
            print(user_text)
            print(ai_text)
            return jsonify({'user_text': user_text, 'ai_text': ai_text})
        
        # Check if a PDF file is uploaded
        elif 'pdf_file' in request.files:
            print(request.files)
            pdf_file = request.files['pdf_file']
            print(pdf_file)
            if pdf_file.filename != '':
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], pdf_file.filename)
                pdf_file.save(file_path)
                
                # Extract text from the uploaded PDF
                user_text = extract_text(file_path)
                if not user_text.strip():  # Handle case where no text is extracted
                    return jsonify({'error': 'No text could be extracted from the PDF'}), 400
                print(user_text)
                ai_text = as_summary(user_text)
                return jsonify({'user_text': 'Processing PDF', 'ai_text': ai_text})
        
        # If neither text nor PDF is provided
        else:
            return jsonify({'error': 'No input provided'}), 400
    
    return render_template('base.html')

def as_summary(txt):
    """
    Summarize the given text using the Hugging Face summarization pipeline.
    """
    try:
        summary = summarizer(txt, max_length=150, min_length=30, do_sample=False)
        out_text = summary[0]['summary_text']
        print(out_text)
        return out_text
    except Exception as e:
        return f"Error during summarization: {str(e)}"

if __name__ == '__main__':
    # Bind to port from environment variable for Render
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)