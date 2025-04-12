from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai
import numpy as np
import re
import os
import nltk
import zipfile
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')
# print("Checking NLTK wordnet resource...")
# wordnet_zip = '/kaggle/working/nltk_data/corpora/wordnet.zip'
# if os.path.exists(wordnet_zip):
#     print(f"Unzipping wordnet from {wordnet_zip}...")
#     with zipfile.ZipFile(wordnet_zip, 'r') as zip_ref:
#         zip_ref.extractall('/kaggle/working/nltk_data/corpora')
#     os.remove(wordnet_zip)
#     print("Wordnet unzipped and zip file removed.")
# else:
#     print("Wordnet already unzipped or not downloaded as zip.")
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import custom_object_scope
import pickle
from newsapi import NewsApiClient
import requests
from bs4 import BeautifulSoup
import string

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app, resources={r"/*": {"origins": "*"}})

# Gemini API for summarization
GEMINI_API_KEY = "AIzaSyCiTHWqSruT06vUTHpqISATkoekBtclPDQ"
genai.configure(api_key=GEMINI_API_KEY)

# NewsAPI key
NEWS_API_KEY = '20a033afa85e4b72af903562634d7f6d'

# Load model and tools
class Cast(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Cast, self).__init__(**kwargs)
    def call(self, inputs):
        return tf.cast(inputs, dtype=self.dtype_policy.compute_dtype)
    def get_config(self):
        return super(Cast, self).get_config()
    @classmethod
    def from_config(cls, config):
        return cls(**config)

def load_file(file_path, load_func, desc):
    if os.path.exists(file_path):
        try:
            return load_func(file_path)
        except Exception as e:
            print(f"Error loading {desc}: {e}")
            return None
    print(f"{desc} not found at {file_path}.")
    return None

model_path = r'C:\Users\SJ\Documents\proj\output\lstm_model.h5'
tokenizer_path = r'C:\Users\SJ\Documents\proj\output\tokenizer.pkl'
scaler_path = r'C:\Users\SJ\Documents\proj\output\scaler.pkl'

with custom_object_scope({'Cast': Cast}):
    model = load_file(model_path, tf.keras.models.load_model, "Model")
tokenizer = load_file(tokenizer_path, lambda p: pickle.load(open(p, 'rb')), "Tokenizer")
scaler = load_file(scaler_path, lambda p: pickle.load(open(p, 'rb')), "Scaler")

if not all([model, tokenizer, scaler]):
    raise SystemExit("Critical files missing. Ensure training script ran.")

# Constants
MAX_SEQUENCE_LENGTH = 100
NUM_FEATURES = 10
stop_words = set(nltk.corpus.stopwords.words('english')) - {'not'}
lemmatizer = nltk.stem.WordNetLemmatizer()
sia = nltk.sentiment.vader.SentimentIntensityAnalyzer()

# Prediction functions
def preprocess_text(text):
    if not isinstance(text, str) or not text.strip(): return ""
    text = re.sub(r"[^a-zA-Z\s]", " ", text.lower())
    words = text.split()
    return " ".join(lemmatizer.lemmatize(word) for word in words if word not in stop_words or word == 'not')

def extract_numerical_features(text):
    if not isinstance(text, str): return np.zeros(NUM_FEATURES)
    words = text.split()
    title = text[:50]
    return np.array([
        len(title.split()), len(words), len(title), len(text),
        sum(1 for c in title if c.isupper()) / len(title) if len(title) > 0 else 0,
        sum(1 for c in text if c.isupper()) / len(text) if len(text) > 0 else 0,
        sum(1 for c in title if c in string.punctuation), sum(1 for c in text if c in string.punctuation),
        sia.polarity_scores(title)['compound'], sia.polarity_scores(text)['compound']
    ])

def predict_batch(texts):
    processed_texts = [preprocess_text(t) for t in texts]
    seqs = tokenizer.texts_to_sequences(processed_texts)
    padded_seqs = pad_sequences(seqs, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
    num_features = scaler.transform(np.array([extract_numerical_features(t) for t in texts]))
    with tf.device('/GPU:0'):
        preds = model.predict([padded_seqs, num_features], batch_size=256, verbose=0)
    return (preds < 0.5).astype(int).flatten()  # 1 = Real, 0 = Fake

# Fetch news
def fetch_news(topic):
    newsapi = NewsApiClient(api_key=NEWS_API_KEY)
    try:
        response = newsapi.get_everything(q=topic, language='en', page_size=5)  # Reduced to 5 for speed
        articles = []
        for article in response['articles']:
            try:
                resp = requests.get(article['url'], timeout=5)
                soup = BeautifulSoup(resp.text, 'html.parser')
                text = " ".join(p.get_text() for p in soup.find_all('p'))
                if text.strip():
                    articles.append(f"{article['title']} {text}")
            except requests.RequestException:
                continue
        return articles
    except Exception as e:
        print(f"Error fetching news: {e}")
        return []

# Summarize endpoint
def summarize_article(article_text):
    try:
        prompt = f"Summarize the following news article in under 100 words:\n\n{article_text}"
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error summarizing article: {str(e)}"

@app.route('/summarize', methods=['POST'])
def summarize():
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({"error": "No article text provided"}), 400
        article_text = data['text'].strip()
        if not article_text:
            return jsonify({"error": "Article text is empty"}), 400
        summary = summarize_article(article_text)
        return jsonify({"summary": summary}), 200
    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500

# Predict endpoint for text input
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({"error": "No text provided"}), 400
        text = data['text'].strip()
        if not text:
            return jsonify({"error": "Text is empty"}), 400
        label = predict_batch([text])[0]
        result = "Real" if label else "Fake"
        return jsonify({"prediction": result}), 200
    except Exception as e:
        return jsonify({"error": f"Prediction error: {str(e)}"}), 500

# Fetch and predict endpoint for topics
@app.route('/fetch_predict', methods=['POST'])
def fetch_predict():
    try:
        data = request.get_json()
        if not data or 'topic' not in data:
            return jsonify({"error": "No topic provided"}), 400
        topic = data['topic'].strip()
        if not topic:
            return jsonify({"error": "Topic is empty"}), 400
        articles = fetch_news(topic)
        if not articles:
            return jsonify({"error": "No articles fetched for this topic"}), 404
        labels = predict_batch(articles)
        results = [{"text": text[:50] + "...", "prediction": "Real" if label else "Fake"}
                   for text, label in zip(articles, labels)]
        return jsonify({"results": results}), 200
    except Exception as e:
        return jsonify({"error": f"Fetch/predict error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)