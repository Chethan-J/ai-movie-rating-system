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
    page_icon="🎬",
    layout="wide"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>
.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: 700;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #777;
    margin-bottom: 30px;
}

.movie-card {
    padding: 20px;
    border-radius: 15px;
    border: 1px solid rgba(128,128,128,0.25);
    margin-bottom: 25px;
}

.rating-card {
    padding: 18px;
    border-radius: 12px;
    border: 1px solid rgba(128,128,128,0.25);
    margin-bottom: 12px;
}

.overall-rating {
    text-align: center;
    padding: 25px;
    border-radius: 18px;
    border: 2px solid rgba(128,128,128,0.25);
    margin-top: 25px;
}

.overall-number {
    font-size: 45px;
    font-weight: 700;
}

.tmdb-note {
    text-align: center;
    color: #777;
    font-size: 13px;
    margin-top: 30px;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🎬 AI Movie Rating System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-powered movie review analysis using BiLSTM and NLP'
    '</div>',
    unsafe_allow_html=True
)

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

st.subheader("🔎 Find a Movie")

movie_name = st.text_input(
    "Movie Name",
    placeholder="Example: Interstellar",
    label_visibility="collapsed"
)

movie = None

if movie_name:

    movie = search_movie(movie_name)

    if movie:

        st.markdown(
            '<div class="movie-card">',
            unsafe_allow_html=True
        )

        col1, col2 = st.columns([1, 2])

        # Movie poster
        if movie.get("poster_path"):

            poster_url = (
                "https://image.tmdb.org/t/p/w500"
                + movie["poster_path"]
            )

            with col1:
                st.image(
                    poster_url,
                    width=280
                )

        # Movie information
        with col2:

            st.markdown(
                f"## {movie.get('title', 'Movie')}"
            )

            release_date = movie.get(
                "release_date",
                ""
            )

            if release_date:

                st.write(
                    f"📅 **Release Date:** {release_date}"
                )

            overview = movie.get(
                "overview",
                ""
            )

            if overview:

                st.write(
                    overview
                )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )

    else:

        st.warning(
            "Movie not found. Try another movie name."
        )

# ============================================================
# REVIEWS
# ============================================================

st.subheader("📝 Analyze Movie Reviews")

st.write(
    "Enter one or more reviews to generate "
    "individual and overall ratings."
)

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
# ANALYZE BUTTON
# ============================================================

if st.button(
    "⭐ Analyze Reviews",
    use_container_width=True
):

    valid_reviews = [
        review
        for review in reviews
        if review.strip() != ""
    ]

    if len(valid_reviews) == 0:

        st.warning(
            "Please enter at least one review."
        )

    else:

        ratings = []

        st.divider()

        st.subheader(
            "📊 Individual Review Ratings"
        )

        # ====================================================
        # INDIVIDUAL PREDICTIONS
        # ====================================================

        for i, review in enumerate(valid_reviews):

            cleaned_review = clean_text(
                review
            )

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
                np.clip(
                    prediction[0][0],
                    1,
                    10
                )
            )

            star_rating = (
                round(predicted_score) / 2
            )

            star_rating = max(
                0.5,
                min(5.0, star_rating)
            )

            ratings.append(
                star_rating
            )

            st.markdown(
                f"""
                <div class="rating-card">
                    <b>Review {i + 1}</b>
                    <br><br>
                    ⭐ <strong>{star_rating:.1f} / 5</strong>
                    <br>
                    IMDb Score: {predicted_score:.2f} / 10
                </div>
                """,
                unsafe_allow_html=True
            )

        # ====================================================
        # OVERALL RATING
        # ====================================================

        overall_rating = (
            sum(ratings) / len(ratings)
        )

        overall_rating = (
            round(overall_rating * 2) / 2
        )

        st.markdown(
            f"""
            <div class="overall-rating">
                <h2>🏆 Overall Movie Rating</h2>
                <div class="overall-number">
                    ⭐ {overall_rating:.1f} / 5
                </div>
                <p>
                    Based on {len(valid_reviews)} review(s)
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

# ============================================================
# TMDB ATTRIBUTION
# ============================================================

st.markdown(
    """
    <div class="tmdb-note">
        This product uses the TMDB API but is not endorsed
        or certified by TMDB.
    </div>
    """,
    unsafe_allow_html=True
)
