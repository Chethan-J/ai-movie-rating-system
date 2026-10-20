import streamlit as st
import tensorflow as tf
import pickle
import re
import requests

st.set_page_config(
    page_title="AI Movie Rating System",
    page_icon="🎬",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0b1020, #151b35);
    color: white;
}

.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    color: #b8c1ec;
    font-size: 18px;
    margin-bottom: 30px;
}

.movie-card {
    padding: 20px;
    border-radius: 15px;
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.1);
}

.rating-box {
    padding: 25px;
    border-radius: 15px;
    background: rgba(255,255,255,0.06);
    text-align: center;
}

.stars {
    font-size: 35px;
}
</style>
""", unsafe_allow_html=True)


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


def clean_text(text):
    text = text.lower()
    text = re.sub(r"<br />", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


TMDB_API_KEY = st.secrets.get("TMDB_API_KEY", "")


def search_movies(movie_name):
    if not TMDB_API_KEY:
        return []

    url = "https://api.themoviedb.org/3/search/movie"

    params = {
        "api_key": TMDB_API_KEY,
        "query": movie_name,
        "language": "en-US",
        "include_adult": False
    }

    try:
        response = requests.get(
            url,
            params=params,
            timeout=10
        )

        if response.status_code == 200:
            return response.json().get(
                "results",
                []
            )

    except:
        return []

    return []


def predict_rating(review):
    cleaned = clean_text(review)

    sequence = tokenizer.texts_to_sequences(
        [cleaned]
    )

    padded = tf.keras.preprocessing.sequence.pad_sequences(
        sequence,
        maxlen=300,
        padding="post",
        truncating="post"
    )

    prediction = model.predict(
        padded,
        verbose=0
    )

    rating = float(prediction[0][0])

    rating = max(
        1.0,
        min(5.0, rating)
    )

    return rating


def get_stars(rating):
    rounded_rating = round(
        rating * 2
    ) / 2

    full_stars = int(
        rounded_rating
    )

    half_star = (
        rounded_rating - full_stars == 0.5
    )

    stars = "⭐" * full_stars

    if half_star:
        stars += "½"

    return stars


st.markdown(
    '<div class="main-title">🎬 AI Movie Rating System</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Predict movie ratings using BiLSTM and NLP'
    '</div>',
    unsafe_allow_html=True
)


st.subheader("🎥 Search for a Movie")

movie_name = st.text_input(
    "Enter movie name",
    placeholder="Example: Bahubali"
)

selected_movie = None


if movie_name:
    movies = search_movies(movie_name)

    if movies:
        st.write("### Select a movie")

        display_movies = movies[:5]

        columns = st.columns(
            len(display_movies)
        )

        for i, movie in enumerate(display_movies):

            with columns[i]:

                poster_path = movie.get(
                    "poster_path"
                )

                if poster_path:
                    poster_url = (
                        "https://image.tmdb.org/t/p/w500"
                        + poster_path
                    )

                    st.image(
                        poster_url,
                        use_container_width=True
                    )

                title = movie.get(
                    "title",
                    "Unknown"
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
                    f"Year: {year}"
                )

                if st.button(
                    "Select Movie",
                    key=f"movie_{movie.get('id')}"
                ):
                    st.session_state.selected_movie = movie

    else:
        st.warning(
            "Movie not found. Try another movie name."
        )


if "selected_movie" in st.session_state:

    selected_movie = (
        st.session_state.selected_movie
    )

    st.markdown("---")
    st.subheader("🎞️ Selected Movie")

    col1, col2 = st.columns(
        [1, 2]
    )

    with col1:

        poster_path = selected_movie.get(
            "poster_path"
        )

        if poster_path:

            poster_url = (
                "https://image.tmdb.org/t/p/w500"
                + poster_path
            )

            st.image(
                poster_url,
                use_container_width=True
            )

    with col2:

        st.markdown(
            f"## {selected_movie.get('title', 'Unknown')}"
        )

        release_date = selected_movie.get(
            "release_date",
            ""
        )

        if release_date:

            st.write(
                f"📅 Release Date: {release_date}"
            )

        tmdb_rating = selected_movie.get(
            "vote_average"
        )

        if tmdb_rating:

            st.write(
                f"⭐ TMDB Rating: {tmdb_rating:.1f}/10"
            )

        overview = selected_movie.get(
            "overview",
            "No overview available."
        )

        st.write(
            f"**Overview:** {overview}"
        )


st.markdown("---")
st.subheader("📝 Enter Movie Reviews")


num_reviews = st.number_input(
    "Number of reviews",
    min_value=1,
    max_value=10,
    value=3,
    step=1
)


reviews = []


for i in range(num_reviews):

    review = st.text_area(
        f"Review {i + 1}",
        placeholder="Write your movie review here...",
        height=120,
        key=f"review_{i}"
    )

    reviews.append(review)


if st.button(
    "🚀 Predict Movie Rating",
    type="primary",
    use_container_width=True
):

    valid_reviews = [
        review.strip()
        for review in reviews
        if review.strip()
    ]

    if not valid_reviews:

        st.warning(
            "Please enter at least one review."
        )

    else:

        predictions = []

        for review in valid_reviews:

            rating = predict_rating(review)

            predictions.append(rating)


        st.markdown("---")

        st.subheader(
            "⭐ Individual Review Ratings"
        )


        result_columns = st.columns(
            min(3, len(predictions))
        )


        for i, rating in enumerate(
            predictions
        ):

            with result_columns[
                i % len(result_columns)
            ]:

                display_rating = (
                    round(rating * 2) / 2
                )

                stars = get_stars(
                    display_rating
                )

                st.markdown(
                    f"### Review {i + 1}"
                )

                st.markdown(
                    f"## {stars}"
                )

                st.markdown(
                    f"### {display_rating:.1f} / 5"
                )

                st.caption(
                    "AI predicted rating"
                )


        overall_rating = sum(
            predictions
        ) / len(predictions)


        overall_rating = (
            round(overall_rating * 2) / 2
        )


        overall_stars = get_stars(
            overall_rating
        )


        st.markdown("---")


        st.markdown(
            '<div class="rating-box">',
            unsafe_allow_html=True
        )

        st.markdown(
            "## 🎯 Overall AI Rating"
        )

        st.markdown(
            f'<div class="stars">{overall_stars}</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f"# {overall_rating:.1f} / 5"
        )

        st.caption(
            f"Based on {len(predictions)} review(s)"
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )


st.markdown("---")

st.caption(
    "This product uses the TMDB API but is not endorsed "
    "or certified by TMDB."
)
