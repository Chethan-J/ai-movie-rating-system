import streamlit as st
import tensorflow as tf
import pickle
import re
import numpy as np
import requests
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Movie Rating System",
    page_icon="🎬"
)

st.title("🎬 AI Movie Rating System")
st.write("Search for a movie and analyze multiple reviews.")

# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("movie_rating_bilstm.keras")

@st.cache_resource
def load_tokenizer():
    with open("movie_rating_tokenizer.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()
tokenizer = load_tokenizer()

# ============================================================
# TMDB MOVIE SEARCH
# ============================================================

def search_movie(movie_name):
    api_key = st.secrets["TMDB_API_KEY"]

    url = "https://api.themoviedb.org/3/search/movie"

    params = {
        "api_key": api_key,
        "query": movie_name
    }

    response = requests.get(url, params=params)

    if response.status_code == 200:
        data = response.json()

        if data["results"]:
            return data["results"][0]

    return None

# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):
    text = text.lower()
    text = re.sub(r"<br />", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# ============================================================
# MOVIE SEARCH
# ============================================================

movie_name = st.text_input(
    "🎬 Enter Movie Name",
    placeholder="Example: Interstellar"
)

if movie_name:

    movie = search_movie(movie_name)

    if movie:

        col1, col2 = st.columns([1, 2])

        if movie.get("poster_path"):

            poster_url = (
                "https://image.tmdb.org/t/p/w500"
                + movie["poster_path"]
            )

            with col1:
                st.image(
                    poster_url,
                    width=250
                )

        with col2:

            st.subheader(movie.get("title", "Movie"))

            release_date = movie.get("release_date", "")

            if release_date:
                st.write(
                    f"📅 Release Date: {release_date}"
                )

            overview = movie.get("overview", "")

            if overview:
                st.write(overview)

    else:
        st.warning("Movie not found. Try another movie name.")

# ============================================================
# MULTIPLE REVIEWS
# ============================================================

st.subheader("📝 Enter Movie Reviews")

review_count = st.number_input(
    "Number of Reviews",
    min_value=1,
    max_value=10,
    value=3,
    step=1
)

reviews = []

for i in range(review_count):

    review = st.text_area(
        f"Review {i + 1}",
        placeholder="Enter your movie review...",
        key=f"review_{i}"
    )

    reviews.append(review)

# ============================================================
# PREDICT RATINGS
# ============================================================

if st.button("⭐ Analyze Reviews"):

    valid_reviews = [
        review for review in reviews
        if review.strip() != ""
    ]

    if len(valid_reviews) == 0:

        st.warning("Please enter at least one review.")

    else:

        ratings = []

        st.subheader("📊 Individual Review Ratings")

        for i, review in enumerate(valid_reviews):

            cleaned_review = clean_text(review)

            sequence = tokenizer.texts_to_sequences(
                [cleaned_review]
            )

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

            star_rating = round(predicted_score) / 2

            star_rating = max(
                0.5,
                min(5.0, star_rating)
            )

            ratings.append(star_rating)

            st.write(
                f"**Review {i + 1}:** "
                f"⭐ {star_rating:.1f} / 5"
            )

        # ====================================================
        # OVERALL RATING
        # ====================================================

        overall_rating = sum(ratings) / len(ratings)

        overall_rating = round(
            overall_rating * 2
        ) / 2

        st.divider()

        st.subheader("🏆 Overall Movie Rating")

        st.write(
            f"# ⭐ {overall_rating:.1f} / 5"
        )

        st.write(
            f"Based on {len(valid_reviews)} review(s)"
        )

# ============================================================
# TMDB ATTRIBUTION
# ============================================================

st.divider()

st.caption(
    "This product uses the TMDB API but is not endorsed "
    "or certified by TMDB."
)
