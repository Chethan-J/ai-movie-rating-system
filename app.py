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

/* ==================== BACKGROUND ==================== */

.stApp {
    background: linear-gradient(
        135deg,
        #050b18 0%,
        #0b1630 50%,
        #111827 100%
    );
    color: white;
}

[data-testid="stHeader"] {
    background: transparent;
}

/* ==================== TITLE ==================== */

.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: 700;
    margin-top: 10px;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #aab7d4;
    margin-bottom: 30px;
}

/* ==================== MOVIE CARD ==================== */

.movie-card {
    padding: 20px;
    border-radius: 15px;
    border: 1px solid #263b63;
    background: rgba(13, 27, 51, 0.75);
    margin-top: 15px;
    margin-bottom: 30px;
}

/* ==================== REVIEW CARD ==================== */

.rating-card {
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #263b63;
    background: rgba(13, 27, 51, 0.75);
    margin-bottom: 15px;
}

/* ==================== OVERALL CARD ==================== */

.overall-card {
    text-align: center;
    padding: 25px;
    border-radius: 18px;
    border: 2px solid #536dfe;
    background: linear-gradient(
        135deg,
        rgba(27, 52, 100, 0.8),
        rgba(53, 40, 110, 0.8)
    );
    margin-top: 25px;
    margin-bottom: 25px;
}

.overall-rating {
    font-size: 45px;
    font-weight: 700;
}

/* ==================== INPUTS ==================== */

.stTextInput input,
.stTextArea textarea,
.stNumberInput input {
    background-color: #111c33 !important;
    color: white !important;
    border: 1px solid #36527d !important;
    border-radius: 10px !important;
}

.stTextInput input::placeholder,
.stTextArea textarea::placeholder {
    color: #8b98b5 !important;
}

/* ==================== BUTTON ==================== */

.stButton > button {
    width: 100%;
    border-radius: 10px;
    border: none;
    background: linear-gradient(
        90deg,
        #168cff,
        #7654ff
    );
    color: white;
    font-size: 17px;
    font-weight: 600;
    padding: 10px;
}

.stButton > button:hover {
    background: linear-gradient(
        90deg,
        #287cff,
        #8b5cf6
    );
    color: white;
}

/* ==================== DIVIDER ==================== */

hr {
    border-color: #263b63;
}

/* ==================== FOOTER ==================== */

.tmdb-note {
    text-align: center;
    color: #7f8ba3;
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
    return tf.keras.models.load_model(
        "movie_rating_bilstm.keras"
    )

@st.cache_resource
def load_tokenizer():
    with open(
        "movie_rating_tokenizer.pkl",
        "rb"
    ) as f:
        return pickle.load(f)

model = load_model()
tokenizer = load_tokenizer()

# ============================================================
# TMDB SEARCH
# ============================================================

def search_movie(movie_name):

    api_key = st.secrets["TMDB_API_KEY"]

    url = "https://api.themoviedb.org/3/search/movie"

    params = {
        "api_key": api_key,
        "query": movie_name
    }

    response = requests.get(
        url,
        params=params
    )

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

    text = re.sub(
        r"<br />",
        " ",
        text
    )

    text = re.sub(
        r"[^a-zA-Z\s]",
        "",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()

# ============================================================
# MOVIE SEARCH
# ============================================================

st.subheader("🔎 Find a Movie")

search_col, button_col = st.columns(
    [5, 1]
)

with search_col:

    movie_name = st.text_input(
        "Movie Name",
        placeholder="Example: Interstellar",
        label_visibility="collapsed"
    )

with button_col:

    search_clicked = st.button(
        "🔍 Search",
        use_container_width=True
    )

# Store movie result
if "movie" not in st.session_state:
    st.session_state.movie = None

if search_clicked:

    if movie_name.strip() == "":

        st.warning(
            "Please enter a movie name."
        )

    else:

        st.session_state.movie = search_movie(
            movie_name
        )

movie = st.session_state.movie

# ============================================================
# MOVIE INFORMATION
# ============================================================

if movie:

    st.markdown(
        '<div class="movie-card">',
        unsafe_allow_html=True
    )

    poster_col, info_col = st.columns(
        [1, 3]
    )

    # -------------------------
    # POSTER
    # -------------------------

    if movie.get("poster_path"):

        poster_url = (
            "https://image.tmdb.org/t/p/w500"
            + movie["poster_path"]
        )

        with poster_col:

            st.image(
                poster_url,
                width=280
            )

    # -------------------------
    # MOVIE DETAILS
    # -------------------------

    with info_col:

        st.markdown(
            f"## {movie.get('title', 'Movie')}"
        )

        release_date = movie.get(
            "release_date",
            ""
        )

        if release_date:

            st.write(
                f"📅 **Release Date:** "
                f"{release_date}"
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

elif search_clicked:

    st.warning(
        "Movie not found. Try another movie name."
    )

# ============================================================
# REVIEW SECTION
# ============================================================

st.subheader(
    "📝 Analyze Movie Reviews"
)

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

# ============================================================
# REVIEW INPUTS — SIDE BY SIDE
# ============================================================

reviews = []

review_count = int(review_count)

# Create rows of 3 reviews
for start in range(
    0,
    review_count,
    3
):

    current_count = min(
        3,
        review_count - start
    )

    columns = st.columns(
        current_count
    )

    for j in range(
        current_count
    ):

        index = start + j

        with columns[j]:

            review = st.text_area(
                f"Review {index + 1}",
                placeholder=(
                    "Enter your movie review..."
                ),
                height=150,
                key=f"review_{index}"
            )

            reviews.append(
                review
            )

# ============================================================
# ANALYZE BUTTON
# ============================================================

st.write("")

analyze_clicked = st.button(
    "⭐ Analyze Reviews",
    use_container_width=True
)

# ============================================================
# PREDICTION
# ============================================================

if analyze_clicked:

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
        scores = []

        st.divider()

        st.subheader(
            "📊 Individual Review Ratings"
        )

        # ====================================================
        # PREDICT EACH REVIEW
        # ====================================================

        for review in valid_reviews:

            cleaned_review = clean_text(
                review
            )

            sequence = (
                tokenizer.texts_to_sequences(
                    [cleaned_review]
                )
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
                min(
                    5.0,
                    star_rating
                )
            )

            ratings.append(
                star_rating
            )

            scores.append(
                predicted_score
            )

        # ====================================================
        # DISPLAY RATINGS SIDE BY SIDE
        # ====================================================

        result_columns = st.columns(
            len(valid_reviews)
            if len(valid_reviews) <= 3
            else 3
        )

        for i, rating in enumerate(
            ratings
        ):

            column_index = i % 3

            with result_columns[
                column_index
            ]:

                st.markdown(
                    f"""
                    <div class="rating-card">

                    <strong>
                    Review {i + 1}
                    </strong>

                    <br><br>

                    ⭐ <strong>
                    {rating:.1f} / 5
                    </strong>

                    <br>

                    <span style="color:#9caac4;">
                    IMDb Score:
                    {scores[i]:.2f} / 10
                    </span>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

        # ====================================================
        # OVERALL RATING
        # ====================================================

        overall_rating = (
            sum(ratings)
            / len(ratings)
        )

        overall_rating = (
            round(
                overall_rating * 2
            ) / 2
        )

        st.markdown(
            '<div class="overall-card">',
            unsafe_allow_html=True
        )

        st.markdown(
            "### 🏆 Overall Movie Rating"
        )

        st.markdown(
            f'<div class="overall-rating">'
            f'⭐ {overall_rating:.1f} / 5'
            f'</div>',
            unsafe_allow_html=True
        )

        st.write(
            f"Based on {len(valid_reviews)} review(s)"
        )

        st.markdown(
            '</div>',
            unsafe_allow_html=True
        )

# ============================================================
# TMDB ATTRIBUTION
# ============================================================

st.markdown(
    '<div class="tmdb-note">'
    'This product uses the TMDB API but is not endorsed '
    'or certified by TMDB.'
    '</div>',
    unsafe_allow_html=True
)
