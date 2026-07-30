import streamlit as st
import pickle
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Text preprocessing
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

# Streamlit UI
st.set_page_config(page_title="Sentiment Analysis", page_icon="😊")

st.title("😊 Sentiment Analysis")
st.write("Enter a sentence to predict its sentiment.")

user_input = st.text_area("Enter Text")

if st.button("Predict"):

    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        cleaned = clean_text(user_input)
        vector = vectorizer.transform([cleaned])
        prediction = model.predict(vector)[0]

        if prediction == "Positive":
            st.success("😊 Positive")

        elif prediction == "Negative":
            st.error("😞 Negative")

        else:
            st.info("😐 Neutral")