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
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("movie_rating_bilstm.keras")

# Load tokenizer
@st.cache_resource
def load_tokenizer():
    with open("movie_rating_tokenizer.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()
tokenizer = load_tokenizer()

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
        # Clean review
        cleaned_review = clean_text(review)

        # Convert text to sequence
        sequence = tokenizer.texts_to_sequences([cleaned_review])

        # Padding
        padded_review = pad_sequences(
            sequence,
            maxlen=200,
            padding="post",
            truncating="post"
        )

        # Predict IMDb score
        prediction = model.predict(
            padded_review,
            verbose=0
        )

        predicted_score = float(
            np.clip(prediction[0][0], 1, 10)
        )

        # Convert IMDb score to nearest 0.5 star
        star_rating = round(predicted_score / 2 * 2) / 2

        # Keep rating between 0.5 and 5
        star_rating = max(0.5, min(5.0, star_rating))

        # Display result
        st.subheader("Prediction")

        st.write(
            f"**Predicted IMDb Score:** {predicted_score:.2f} / 10"
        )

        st.write(
            f"**Movie Rating:** ⭐ {star_rating:.1f} / 5"
        )
