import streamlit as st
import pandas as pd
import pickle
import re
import string
import nltk

# Download NLTK data (only first time)
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ----------------------------
# Load Model & Vectorizer
# ----------------------------
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# ----------------------------
# Text Preprocessing
# ----------------------------
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)

    words = []
    for word in text.split():
        if word not in stop_words:
            words.append(lemmatizer.lemmatize(word))

    return " ".join(words)

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(
    page_title="Sentiment Analysis",
    page_icon="😊",
    layout="centered"
)

st.title("😊 Sentiment Analysis App")

st.write("Type any sentence below and click **Predict**.")

user_input = st.text_area(
    "Enter your text",
    height=150,
    placeholder="Example: I really love this phone"
)

if st.button("Predict Sentiment"):

    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:

        cleaned_text = clean_text(user_input)

        vector = vectorizer.transform([cleaned_text])

        prediction = model.predict(vector)[0]

        st.subheader("Prediction")

        if prediction.lower() == "positive":
            st.success("😊 Positive")

        elif prediction.lower() == "negative":
            st.error("😞 Negative")

        else:
            st.info("😐 Neutral")