import streamlit as st
import pickle
import re
import string

# Load model and vectorizer
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Simple text cleaning (No NLTK)
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = " ".join(text.split())
    return text

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

        st.subheader("Prediction")

        if prediction == "Positive":
            st.success("😊 Positive")

        elif prediction == "Negative":
            st.error("😞 Negative")

        else:
            st.info("😐 Neutral")