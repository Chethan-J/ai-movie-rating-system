import streamlit as st
import tensorflow as tf
import pickle
import re
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Page configuration
st.set_page_config(
    page_title="AI Movie Rating System",
    page_icon="🎬"
)

st.title("🎬 AI Movie Rating System")
st.write("Enter a movie review to predict its rating.")

# Load model
model = tf.keras.models.load_model("movie_rating_bilstm.keras")

# Load tokenizer
with open("movie_rating_tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Text cleaning
def clean_text(text):
    text = text.lower()
    text = re.sub(r"<br />", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# Review input
review = st.text_area(
    "Enter your movie review:",
    placeholder="Example: This movie was amazing and the acting was excellent!"
)

# Prediction
if st.button("Predict Rating"):

    if review.strip() == "":
        st.warning("Please enter a review.")

    else:
        cleaned_review = clean_text(review)

        sequence = tokenizer.texts_to_sequences([cleaned_review])

        padded_review = pad_sequences(
            sequence,
            maxlen=200,
            padding="post",
            truncating="post"
        )

        prediction = model.predict(
            padded_review,
            verbose=0
        )

        predicted_score = float(
            np.clip(prediction[0][0], 1, 10)
        )

        # Convert 1–10 score to nearest 0.5 star
        star_rating = round(predicted_score * 2) / 4

        st.subheader("Prediction")

        st.write(
            f"**Predicted IMDb Score:** {predicted_score:.2f} / 10"
        )

        st.write(
            f"**Movie Rating:** ⭐ {star_rating:.1f} / 5"
        )
