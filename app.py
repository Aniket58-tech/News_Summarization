import streamlit as st
import joblib
import nltk
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os

# Create a folder to store nltk data in the app directory
nltk_data_dir = os.path.join(os.getcwd(), "nltk_data")
if not os.path.exists(nltk_data_dir):
    os.makedirs(nltk_data_dir)

# Download necessary resources into this local folder
for pkg in ["punkt", "punkt_tab"]:
    nltk.download(pkg, download_dir=nltk_data_dir)

# Tell NLTK to use this folder
nltk.data.path.append(nltk_data_dir)

# --------------------
# Load trained classifier and vectorizer
# --------------------
model = joblib.load("best_news_classifier.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")  # same vectorizer used during training

# --------------------
# Summarization function
# --------------------
def summarize_text(processed_text, top_n=3):
    sentences = sent_tokenize(processed_text)
    if len(sentences) <= top_n:
        return processed_text
    
    # TF-IDF per sentence (local, only for summarization)
    tfidf = TfidfVectorizer(stop_words='english')
    sentence_vectors = tfidf.fit_transform(sentences)
    
    # Cosine similarity
    sim_matrix = cosine_similarity(sentence_vectors)
    
    # Score sentences by sum of similarities
    scores = sim_matrix.sum(axis=1)
    
    # Rank sentences and pick top_n
    ranked_sentences = [sentences[i] for i in np.argsort(-scores)]
    return " ".join(ranked_sentences[:top_n])

# --------------------
# Streamlit UI
# --------------------
st.title("News Categorization & Summarization Demo")

user_input = st.text_area("Enter a news article here:", height=200)

top_n = st.slider("Number of sentences in summary:", min_value=1, max_value=5, value=3)

if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("Please enter some text!")
    else:
        # Preprocessing (simple lowercase; apply more if needed)
        processed_input = user_input.lower()
        
        # 1️⃣ Predict category
        X_input = vectorizer.transform([processed_input])
        category = model.predict(X_input)[0]
        st.subheader("Predicted Category:")
        st.success(category)
        
        # 2️⃣ Generate summary
        summary = summarize_text(processed_input, top_n=top_n)
        st.subheader("Summary:")
        st.write(summary)
