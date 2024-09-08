from flask import Flask, request, render_template
import pickle
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

app = Flask(__name__)

# Ensure NLTK data is available
def ensure_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

ensure_nltk_data()

# Load model and vectorizer
model_path = 'model.pkl'
vectorizer_path = 'vectorizer.pkl'

try:
    model = pickle.load(open(model_path, 'rb'))
    tfidf = pickle.load(open(vectorizer_path, 'rb'))
except FileNotFoundError as e:
    raise RuntimeError(f"Model or vectorizer file not found: {e}")

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = [i for i in text if i.isalnum()]
    y = [ps.stem(i) for i in y]
    return " ".join(y)

@app.route('/', methods=['GET', 'POST'])
def home():
    result_text = None
    try:
        if request.method == 'POST':
            email_text = request.form['text']
            if email_text.strip():  # Check if text is not empty
                transformed_text = transform_text(email_text)
                vector_input = tfidf.transform([transformed_text])
                result = model.predict(vector_input)[0]
                result_text = 'Spam' if result == 1 else 'Not Spam'
    except Exception as e:
        result_text = f"An error occurred: {str(e)}"
    return render_template('index.html', result=result_text)

if __name__ == '__main__':
    app.run(debug=True)
