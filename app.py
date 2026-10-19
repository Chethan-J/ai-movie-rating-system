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
# CUSTOM UI
# ============================================================

st.markdown("""
<style>

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

.main-title {
    font-size: 42px;
    font-weight: 800;
    text-align: center;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    color: #aab8d4;
    font-size: 17px;
    margin-bottom: 30px;
}

.movie-card {
    padding: 20px;
    border-radius: 15px;
    border: 1px solid #263b63;
    background: rgba(13, 27, 51, 0.75);
    margin-top: 15px;
    margin-bottom: 30px;
}

.movie-option {
    padding: 12px;
    border-radius: 15px;
    border: 1px solid #263b63;
    background: rgba(13, 27, 51, 0.75);
    min-height: 380px;
}

.rating-card {
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #263b63;
    background: rgba(13, 27, 51, 0.75);
    margin-bottom: 15px;
    min-height: 240px;
}

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

.tmdb-note {
    text-align: center;
    color: #8998b5;
    font-size: 13px;
    margin-top: 30px;
}

.stTextInput input,
.stTextArea textarea,
.stNumberInput input {
    background-color: #111c33 !important;
    color: white !important;
    border: 1px solid #36527d !important;
    border-radius: 10px !important;
}

.stButton > button {
    width: 100%;
    border-radius: 10px;
    border: none;
    background: linear-gradient(90deg, #168cff, #7654ff);
    color: white;
    font-size: 16px;
    font-weight: 600;
    padding: 10px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="main-title">🎬 AI Movie Rating System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Analyze movie reviews using NLP and BiLSTM to predict ratings'
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
# TEXT CLEANING
# ============================================================

def clean_text(text):
    text = text.lower()
    text = re.sub(r"<br />", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ============================================================
# TMDB SEARCH
# ============================================================

def tmdb_search(query, page=1):
    api_key = st.secrets["TMDB_API_KEY"]

    url = "https://api.themoviedb.org/3/search/movie"

    params = {
        "api_key": api_key,
        "query": query,
        "page": page,
        "include_adult": False
    }

    try:
        response = requests.get(
            url,
            params=params,
            timeout=10
        )

        if response.status_code == 200:
            return response.json()

    except Exception:
        return None

    return None


# ============================================================
# GET MOVIE SEARCH RESULTS
# ============================================================

def search_movies(movie_name):

    movie_name = movie_name.strip()

    if not movie_name:
        return []

    candidates = {}

    # --------------------------------------------------------
    # 1. Search the complete phrase
    # --------------------------------------------------------

    data = tmdb_search(movie_name)

    if data:
        for movie in data.get("results", [])[:10]:

            movie_id = movie.get("id")

            if movie_id:
                candidates[movie_id] = movie


    # --------------------------------------------------------
    # 2. Search individual words
    # --------------------------------------------------------

    words = movie_name.split()

    for word in words:

        # Ignore extremely short words
        if len(word) < 3:
            continue

        data = tmdb_search(word)

        if data:

            for movie in data.get("results", [])[:10]:

                movie_id = movie.get("id")

                if movie_id:
                    candidates[movie_id] = movie


    # --------------------------------------------------------
    # Return results
    # --------------------------------------------------------

    return list(candidates.values())[:10]


# ============================================================
# SESSION STATE
# ============================================================

if "movie" not in st.session_state:
    st.session_state.movie = None

if "search_results" not in st.session_state:
    st.session_state.search_results = []

if "search_performed" not in st.session_state:
    st.session_state.search_performed = False


# ============================================================
# MOVIE SEARCH
# ============================================================

st.subheader("🔎 Find a Movie")

search_col, button_col = st.columns([5, 1])

with search_col:

    movie_name = st.text_input(
        "Movie Name",
        placeholder="Example: Interstellar or raj huli",
        label_visibility="collapsed"
    )

with button_col:

    search_clicked = st.button(
        "🔍 Search",
        use_container_width=True
    )


# ============================================================
# PERFORM SEARCH
# ============================================================

if search_clicked:

    if movie_name.strip() == "":

        st.warning("Please enter a movie name.")

    else:

        results = search_movies(movie_name)

        st.session_state.search_results = results
        st.session_state.search_performed = True
        st.session_state.movie = None


# ============================================================
# SHOW SEARCH RESULTS
# ============================================================

if (
    st.session_state.search_performed
    and st.session_state.movie is None
):

    results = st.session_state.search_results

    if len(results) == 0:

        st.warning(
            "No movies found. Try another movie name."
        )

    else:

        st.markdown("### 🎬 Select Your Movie")

        st.caption(
            "We found these movies related to your search. "
            "Select the movie you want to analyze."
        )

        # Show maximum 5 options at once
        results = results[:5]

        columns = st.columns(len(results))

        for i, movie in enumerate(results):

            with columns[i]:

                poster_path = movie.get("poster_path")

                if poster_path:

                    poster_url = (
                        "https://image.tmdb.org/t/p/w500"
                        + poster_path
                    )

                    st.image(
                        poster_url,
                        use_container_width=True
                    )

                else:

                    st.markdown(
                        """
                        <div style="
                            height:250px;
                            display:flex;
                            align-items:center;
                            justify-content:center;
                            background:#111c33;
                            border-radius:10px;
                            color:#8998b5;
                        ">
                        🎬<br>No Poster
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                title = movie.get(
                    "title",
                    "Unknown Movie"
                )

                release_date = movie.get(
                    "release_date",
                    ""
                )

                year = (
                    release_date[:4]
                    if release_date
                    else "N/A"
                )

                st.markdown(
                    f"**{title}**"
                )

                st.caption(
                    f"📅 {year}"
                )

                if st.button(
                    "Select Movie",
                    key=f"select_movie_{movie.get('id')}",
                    use_container_width=True
                ):

                    st.session_state.movie = movie
                    st.session_state.search_results = []
                    st.rerun()


# ============================================================
# SELECTED MOVIE
# ============================================================

movie = st.session_state.movie


if movie is not None:

    st.markdown(
        '<div class="movie-card">',
        unsafe_allow_html=True
    )

    poster_path = movie.get("poster_path")

    movie_col1, movie_col2 = st.columns([1, 2])

    with movie_col1:

        if poster_path:

            poster_url = (
                "https://image.tmdb.org/t/p/w500"
                + poster_path
            )

            st.image(
                poster_url,
                use_container_width=True
            )

        else:

            st.info("Poster not available.")


    with movie_col2:

        title = movie.get(
            "title",
            "Unknown Movie"
        )

        release_date = movie.get(
            "release_date",
            "Unknown"
        )

        overview = movie.get(
            "overview",
            "No description available."
        )

        rating = movie.get(
            "vote_average",
            0
        )

        st.markdown(
            f"## 🎬 {title}"
        )

        st.markdown(
            f"**Release Date:** {release_date}"
        )

        st.markdown(
            f"**TMDB Rating:** ⭐ {rating}/10"
        )

        st.write(overview)

    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )


    # ========================================================
    # REVIEW INPUT
    # ========================================================

    st.subheader("📝 Enter Movie Reviews")

    review_count = st.number_input(
        "Number of Reviews",
        min_value=1,
        max_value=10,
        value=3,
        step=1
    )

    reviews = []

    for row_start in range(0, review_count, 3):

        row_count = min(
            3,
            review_count - row_start
        )

        cols = st.columns(row_count)

        for i in range(row_count):

            index = row_start + i + 1

            with cols[i]:

                review = st.text_area(
                    f"Review {index}",
                    placeholder=(
                        "Enter your movie review..."
                    ),
                    height=160,
                    key=f"review_{index}"
                )

                reviews.append(review)


    # ========================================================
    # ANALYZE BUTTON
    # ========================================================

    analyze_clicked = st.button(
        "🎯 Analyze Reviews",
        use_container_width=True
    )


    # ========================================================
    # PREDICTIONS
    # ========================================================

    if analyze_clicked:

        valid_reviews = [
            review
            for review in reviews
            if review.strip()
        ]

        if len(valid_reviews) == 0:

            st.warning(
                "Please enter at least one review."
            )

        else:

            predictions = []

            for review in valid_reviews:

                cleaned = clean_text(review)

                sequence = tokenizer.texts_to_sequences(
                    [cleaned]
                )

                padded = pad_sequences(
                    sequence,
                    maxlen=200,
                    padding="post",
                    truncating="post"
                )

                predicted_score = float(
                    model.predict(
                        padded,
                        verbose=0
                    )[0][0]
                )

                # Keep score inside IMDb range
                predicted_score = max(
                    1.0,
                    min(10.0, predicted_score)
                )

                # Convert 1-10 score to
                # half-star 0.5-5.0 rating
                star_rating = round(
                    predicted_score
                ) / 2

                star_rating = max(
                    0.5,
                    min(5.0, star_rating)
                )

                predictions.append(
                    star_rating
                )


            # ====================================================
            # INDIVIDUAL RATINGS
            # ====================================================

            st.subheader(
                "⭐ Individual Review Ratings"
            )

            result_columns = st.columns(
                min(3, len(predictions))
            )

            for i, rating in enumerate(predictions):

                with result_columns[
                    i % len(result_columns)
                ]:

                    st.markdown(
                        f"""
                        <div class="rating-card">
                            <h3>Review {i + 1}</h3>
                            <div style="
                                font-size:35px;
                                margin-top:15px;
                            ">
                                {"⭐" * int(rating)}
                            </div>
                            <h2>{rating:.1f} / 5</h2>
                            <p>
                                AI predicted rating
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )


            # ====================================================
            # OVERALL RATING
            # ====================================================

            overall_rating = sum(
                predictions
            ) / len(predictions)

            # Round to nearest half star
            overall_rating = round(
                overall_rating * 2
            ) / 2

            overall_rating = max(
                0.5,
                min(5.0, overall_rating)
            )

            full_stars = int(
                overall_rating
            )

            half_star = (
                overall_rating % 1 == 0.5
            )

            stars = "⭐" * full_stars

            if half_star:

                stars += "✨"


            st.markdown(
                f"""
                <div class="overall-card">

                    <div style="
                        font-size:22px;
                        color:#b7c5e0;
                    ">
                        Overall AI Rating
                    </div>

                    <div class="overall-rating">
                        {overall_rating:.1f} / 5
                    </div>

                    <div style="
                        font-size:30px;
                        margin-top:10px;
                    ">
                        {stars}
                    </div>

                    <div style="
                        margin-top:10px;
                        color:#aab8d4;
                    ">
                        Based on {len(predictions)}
                        review(s)
                    </div>

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
