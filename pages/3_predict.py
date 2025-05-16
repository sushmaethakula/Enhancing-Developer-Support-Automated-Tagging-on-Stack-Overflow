import streamlit as st
import joblib
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from autocorrect import Speller
from contractions import fix
from textacy.preprocessing.remove import accents
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from sklearn.linear_model import SGDClassifier

import nltk
nltk.download('all')
# Inject custom CSS
st.markdown(f"""
    <style>
        .stApp {{
            background: linear-gradient(135deg, #0f0f0f, #3a3a3a); /* Black gradient */
            color: #ffffff;
            position: relative;
            overflow: hidden;
            min-height: 100vh;
            font-family: 'Segoe UI', sans-serif;
        }}

        .tag-container {{
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-top: 10px;
        }}

        .tag {{
            background: #1a1a1a;
            color: #00ffcc;
            padding: 8px 16px;
            border: 2px solid #00ffcc;
            border-radius: 25px;
            font-weight: bold;
            text-shadow: 0 0 5px #00ffcc;
            box-shadow: 0 0 10px #00ffcc inset, 0 0 10px #00ffcc;
            transition: transform 0.2s ease-in-out;
        }}

        .tag:hover {{
            transform: scale(1.1);
        }}

        .stButton>button {{
            background-color: #00ffcc;
            color: black;
            font-weight: bold;
            border-radius: 10px;
            padding: 10px 20px;
            transition: 0.3s ease;
            border: 2px solid #00ffcc;
        }}

        .stButton>button:hover {{
            background-color: #00bfa6;
            color: white;
        }}

        h1 {{
            text-align: center;
            text-shadow: 0 0 10px #00ffcc;
            margin-bottom: 30px;
        }}
    </style>
""", unsafe_allow_html=True)

# Load saved components
lr = joblib.load('SGD_model.pkl')
tfidf = joblib.load('TFIDF.pkl')
mlb_classes = joblib.load('mlb_classes.pkl')

# Initialize preprocessing tools
speller = Speller()
stopword = stopwords.words("english")
stem = SnowballStemmer("english")
lem = WordNetLemmatizer()

# Text preprocessing function
def text_pre_processing(text):
    text = text.lower()
    text = re.sub(r"[^a-z]", " ", text.lower())  # Normalize case & remove punctuation
    text = accents(text)
    words = word_tokenize(text)
    new_text = [lem.lemmatize(stem.stem(word)) for word in words if word not in stopword]
    return " ".join(new_text)

# Prediction function
def predict_tags(text, model=lr, threshold=0.3):
    clean_text = text_pre_processing(text)
    text_tfidf = tfidf.transform([clean_text])

    if hasattr(model, 'predict_proba'):
        probas = model.predict_proba(text_tfidf)
        tags = [mlb_classes[i] for i, class_name in enumerate(mlb_classes) if probas[i][0][1] > threshold]
    else:
        preds = model.predict(text_tfidf)
        tags = [mlb_classes[i] for i, val in enumerate(preds[0]) if val == 1]

    return tags

# App Title
st.title("🌟 Enhancing Developer Support: Automated Tagging on Stack Overflow")

# Input Text Area
test_texts = st.text_area("📝 Submit your Stack Overflow content to generate tags:", height=200, value="Type here........")

# Predict Button
if st.button("✨ Generate Tags"):
    predicted_tags = predict_tags(test_texts)

    st.write("### 🏷️ Predicted Tags:")
    tag_html = '<div class="tag-container">'
    for tag in predicted_tags:
        tag_html += f'<span class="tag">{tag}</span>'
    tag_html += '</div>'
    st.markdown(tag_html, unsafe_allow_html=True)
