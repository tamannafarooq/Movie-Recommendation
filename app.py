import hashlib
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_PATH = Path(__file__).parent / "movies.csv"
ARTIFACT_DIR = Path(__file__).parent / "artifacts"
REQUIRED_COLUMNS = ["title", "genres", "keywords", "overview"]


def load_movies(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df[REQUIRED_COLUMNS].copy()


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()
    for col in REQUIRED_COLUMNS:
        prepared[col] = prepared[col].fillna("").astype(str)

    prepared["tags"] = (
        prepared["genres"]
        + " "
        + prepared["keywords"]
        + " "
        + prepared["overview"]
    )
    prepared["tags"] = (
        prepared["tags"]
        .str.lower()
        .str.replace(r"[^a-z0-9\s]", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    prepared = prepared[prepared["tags"].str.len() > 0].reset_index(drop=True)
    return prepared


def build_similarity_model(df: pd.DataFrame, method: str, max_features: int):
    if method == "CountVectorizer":
        vectorizer = CountVectorizer(max_features=max_features, stop_words="english")
    else:
        vectorizer = TfidfVectorizer(max_features=max_features, stop_words="english")

    vectors = vectorizer.fit_transform(df["tags"]).toarray()
    similarity_matrix = cosine_similarity(vectors)
    title_to_index = pd.Series(df.index, index=df["title"].str.lower()).drop_duplicates()

    return similarity_matrix, title_to_index


def dataset_signature(df: pd.DataFrame) -> str:
    joined = "\n".join(df["tags"].tolist())
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()


def artifact_path(method: str, max_features: int) -> Path:
    method_key = method.lower().replace("-", "_")
    return ARTIFACT_DIR / f"similarity_{method_key}_{max_features}.pkl"


@st.cache_resource
def load_or_train_model(df: pd.DataFrame, method: str, max_features: int):
    ARTIFACT_DIR.mkdir(exist_ok=True)
    signature = dataset_signature(df)
    path = artifact_path(method, max_features)

    if path.exists():
        with path.open("rb") as file:
            payload = pickle.load(file)
        if payload.get("signature") == signature:
            return payload["similarity_matrix"], payload["title_to_index"], True

    similarity_matrix, title_to_index = build_similarity_model(df, method, max_features)
    with path.open("wb") as file:
        pickle.dump(
            {
                "signature": signature,
                "similarity_matrix": similarity_matrix,
                "title_to_index": title_to_index,
            },
            file,
        )

    return similarity_matrix, title_to_index, False


def recommend_movies(
    title: str,
    df: pd.DataFrame,
    similarity_matrix,
    title_to_index: pd.Series,
    top_n: int = 5,
):
    movie_index = title_to_index.get(title.lower())
    if movie_index is None:
        return pd.DataFrame(columns=["title", "score", "genres"])

    distances = list(enumerate(similarity_matrix[movie_index]))
    ranked = sorted(distances, key=lambda x: x[1], reverse=True)
    top_matches = ranked[1 : top_n + 1]

    rows = []
    for idx, score in top_matches:
        rows.append(
            {
                "title": df.iloc[idx]["title"],
                "genres": df.iloc[idx]["genres"],
                "score": round(float(score), 4),
            }
        )

    return pd.DataFrame(rows)


def parse_genres(genres_text: str):
    return {g.strip().lower() for g in genres_text.split() if g.strip()}


def evaluate_recommender(df: pd.DataFrame, similarity_matrix, k: int = 5):
    genre_sets = [parse_genres(g) for g in df["genres"]]
    total = len(df)
    if total == 0:
        return {"genre_match_at_k": 0.0, "catalog_coverage": 0.0, "mean_similarity": 0.0}

    genre_match_scores = []
    all_recommended_idx = set()
    mean_similarity_scores = []

    for idx in range(total):
        distances = list(enumerate(similarity_matrix[idx]))
        ranked = sorted(distances, key=lambda x: x[1], reverse=True)[1 : k + 1]
        rec_indices = [i for i, _ in ranked]

        all_recommended_idx.update(rec_indices)
        mean_similarity_scores.append(float(np.mean([score for _, score in ranked])) if ranked else 0.0)

        seed_genres = genre_sets[idx]
        if not seed_genres:
            genre_match_scores.append(0.0)
            continue

        hits = 0
        for rec_idx in rec_indices:
            if seed_genres.intersection(genre_sets[rec_idx]):
                hits += 1
        genre_match_scores.append(hits / max(k, 1))

    return {
        "genre_match_at_k": float(np.mean(genre_match_scores)),
        "catalog_coverage": len(all_recommended_idx) / total,
        "mean_similarity": float(np.mean(mean_similarity_scores)),
    }


def genre_distribution(df: pd.DataFrame):
    exploded = (
        df["genres"]
        .str.split()
        .explode()
        .dropna()
        .str.strip()
        .str.lower()
    )
    return exploded.value_counts().rename_axis("genre").reset_index(name="count")


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_poster_url(movie_title: str, api_key: str):
    if not api_key:
        return None

    url = "https://api.themoviedb.org/3/search/movie"
    params = {"api_key": api_key, "query": movie_title}
    try:
        response = requests.get(url, params=params, timeout=8)
        response.raise_for_status()
        results = response.json().get("results", [])
        if not results:
            return None

        poster_path = results[0].get("poster_path")
        if not poster_path:
            return None
        return f"https://image.tmdb.org/t/p/w500{poster_path}"
    except requests.RequestException:
        return None


def main():
    st.set_page_config(
        page_title="Movie Recommendation System",
        page_icon="🎬",
        layout="wide",
    )

    # User Information Collection
    if 'user_info' not in st.session_state:
        st.title("🎬 Welcome to Movie Recommendation System")
        st.subheader("Please tell us about yourself to personalize your experience")

        with st.form("user_info_form"):
            col1, col2, col3 = st.columns(3)

            with col1:
                name = st.text_input("Your Name", placeholder="Enter your name")

            with col2:
                gender = st.selectbox("Gender", ["Select", "Male", "Female", "Other", "Prefer not to say"])

            with col3:
                region = st.selectbox("Region", ["Select", "North America", "South America", "Europe", "Asia", "Africa", "Australia/Oceania", "Other"])

            submitted = st.form_submit_button("Continue to App")

            if submitted:
                if name.strip() and gender != "Select" and region != "Select":
                    st.session_state.user_info = {
                        'name': name.strip(),
                        'gender': gender,
                        'region': region
                    }
                    st.success(f"Welcome {name}! Let's find some great movies for you.")
                    st.rerun()
                else:
                    st.error("Please fill in all fields to continue.")

        return  # Don't show the rest of the app until user info is collected

    # Show user greeting in sidebar
    user_info = st.session_state.user_info
    st.sidebar.markdown(f"**👋 Welcome, {user_info['name']}!**")
    st.sidebar.markdown(f"📍 Region: {user_info['region']}")
    st.sidebar.markdown("---")

    st.title("Content-Based Movie Recommendation System")
    st.caption("Built with cosine similarity over text features and optional TMDB poster integration")

    if not DATA_PATH.exists():
        st.error("movies.csv was not found. Place it in the project root and rerun.")
        st.stop()

    data = load_movies(DATA_PATH)
    data = prepare_features(data)

    st.sidebar.header("Model Settings")
    vectorizer_method = st.sidebar.selectbox(
        "Vectorization method",
        ["CountVectorizer", "TF-IDF"],
        index=1,
    )
    max_features = st.sidebar.slider("Max features", min_value=1000, max_value=10000, value=5000, step=500)
    top_n = st.sidebar.slider("Recommendations", min_value=3, max_value=10, value=5, step=1)
    show_posters = st.sidebar.checkbox("Show TMDB posters", value=False)

    default_api_key = os.getenv("TMDB_API_KEY", "")
    if show_posters:
        tmdb_api_key = st.sidebar.text_input("TMDB API key", value=default_api_key, type="password")
        st.sidebar.caption("Create a free key at themoviedb.org")
    else:
        tmdb_api_key = ""

    similarity_matrix, title_to_index, loaded_from_cache = load_or_train_model(
        data,
        vectorizer_method,
        max_features,
    )

    if loaded_from_cache:
        st.sidebar.success("Model loaded from local artifact cache")
    else:
        st.sidebar.info("Model trained and artifact cache updated")

    tab_recommend, tab_evaluate, tab_explore, tab_viva = st.tabs(
        ["Recommend", "Evaluate", "Explore Data", "Viva Notes"]
    )

    with tab_recommend:
        col1, col2 = st.columns([2, 1])
        with col1:
            selected_movie = st.selectbox("Choose a movie", sorted(data["title"].unique()))
            if st.button("Recommend", type="primary"):
                output = recommend_movies(selected_movie, data, similarity_matrix, title_to_index, top_n)
                if output.empty:
                    st.warning("No recommendations found for this movie.")
                else:
                    st.subheader(f"Top {top_n} recommendations for {selected_movie}")
                    st.dataframe(output, use_container_width=True)

                    st.markdown("### Recommendation Cards")
                    card_cols = st.columns(min(3, top_n))
                    for i, (_, row) in enumerate(output.iterrows()):
                        with card_cols[i % len(card_cols)]:
                            poster_url = None
                            if show_posters and tmdb_api_key:
                                poster_url = fetch_poster_url(row["title"], tmdb_api_key)
                            if poster_url:
                                st.image(poster_url, use_container_width=True)
                            st.markdown(f"**{row['title']}**")
                            st.caption(f"Genres: {row['genres']}")
                            st.progress(min(float(row["score"]), 1.0), text=f"Similarity {row['score']}")

                    if show_posters and not tmdb_api_key:
                        st.warning("Add a TMDB API key in the sidebar to fetch posters.")

        with col2:
            st.markdown("### Pipeline")
            st.markdown(
                """
                1. Combine genres, keywords, and overview into a single text feature.
                2. Convert text into vectors with CountVectorizer or TF-IDF.
                3. Compute cosine similarity between movie vectors.
                4. Recommend top-N most similar movies.
                """
            )
            st.info("Tip: Try both vectorizers and compare recommendation quality.")

    with tab_evaluate:
        st.subheader("Offline Evaluation Dashboard")
        st.caption("Proxy metrics to discuss model quality in interviews and viva.")
        k_eval = st.slider("Evaluation K", min_value=3, max_value=10, value=5, step=1, key="k_eval")
        metrics = evaluate_recommender(data, similarity_matrix, k=k_eval)

        m1, m2, m3 = st.columns(3)
        m1.metric("Genre Match@K", f"{metrics['genre_match_at_k']:.3f}")
        m2.metric("Catalog Coverage", f"{metrics['catalog_coverage']:.3f}")
        m3.metric("Mean Similarity", f"{metrics['mean_similarity']:.3f}")

        st.markdown("### Metric Intuition")
        st.markdown(
            """
            - **Genre Match@K**: fraction of recommended movies sharing at least one genre token with the seed movie.
            - **Catalog Coverage**: proportion of the catalog that appears in at least one recommendation list.
            - **Mean Similarity**: average cosine score of recommended items.
            """
        )

    with tab_explore:
        st.subheader("Dataset Explorer")
        st.write(f"Total movies: **{len(data)}**")
        st.dataframe(data[["title", "genres", "keywords"]], use_container_width=True)

        st.markdown("### Genre Distribution")
        genre_df = genre_distribution(data).head(15)
        st.bar_chart(genre_df.set_index("genre"))

        st.markdown("### Sample Overviews")
        sample_df = data[["title", "overview"]].sample(min(5, len(data)), random_state=42)
        st.table(sample_df)

    with tab_viva:
        st.markdown(
            """
            - **Why recommendation systems?** They personalize user experience and improve engagement.
            - **Why cosine similarity?** It compares direction of text vectors and works well for sparse text features.
            - **What is vectorization?** Turning words into numeric features for machine learning.
            - **CountVectorizer vs TF-IDF:** TF-IDF gives higher weight to informative rare terms.
            """
        )

        st.markdown("### Useful Formula")
        st.latex(r"\text{cosine}(A,B) = \frac{A \cdot B}{\|A\|\|B\|}")


if __name__ == "__main__":
    main()
