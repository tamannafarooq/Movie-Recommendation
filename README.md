# Content-Based Recommendation System using Cosine Similarity

A complete standalone ML project that recommends similar movies using text features and cosine similarity, with an interactive Streamlit app.

## Key Features

- Content-based recommendations using cosine similarity
- Improved text preprocessing and unified movie tags from title, genres, keywords, and overview
- Switch between CountVectorizer and TF-IDF from the sidebar
- Local artifact caching for faster repeated runs
- User onboarding form with demographic context displayed in sidebar
- Optional TMDB poster preview cards
- Multi-tab app: Recommend, Evaluate, Explore Data, Viva Notes
- Offline evaluation dashboard (Genre Match@K, Coverage, Mean Similarity)
- Streamlit-ready project structure for local run and cloud deployment

## Objective

Build a recommendation system that:
- Accepts a movie title as input
- Recommends similar movies based on content features

## Tech Stack

- Python
- Pandas
- Scikit-learn
- Streamlit

## Dataset

This project ships with a sample `movies.csv` containing these required columns:
- `title`
- `genres`
- `keywords`
- `overview`

You can replace it with a TMDB-based dataset as long as these columns are available.

## Project Structure

```text
ml/
├── .streamlit/
│   ├── config.toml
│   └── secrets.toml.example
├── artifacts/                  # Auto-created local model cache
├── .gitignore
├── app.py
├── movies.csv
├── requirements.txt
└── README.md
```

## How It Works

1. Combine text columns into one `tags` field.
2. Vectorize `tags` using CountVectorizer or TF-IDF.
3. Compute pairwise cosine similarity.
4. Cache similarity artifacts locally for repeated runs.
5. Return top-N most similar movies.

## App Tabs

- Recommend: choose a movie, generate top-N similar items, view recommendation cards.
- Evaluate: inspect offline proxy metrics for model quality discussion.
- Explore Data: inspect records and genre distribution chart.
- Viva Notes: quick theory, interview points, and cosine similarity formula.

## Setup and Run

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the app:

```bash
streamlit run app.py
```

4. Open the local URL shown in your terminal.

## Optional: Enable TMDB Posters

1. Create a free TMDB API key.
2. Set environment variable before running:

```bash
# PowerShell
$env:TMDB_API_KEY="your_key_here"
```

3. In the app sidebar, enable "Show TMDB posters".

Alternative: for Streamlit Cloud, define `TMDB_API_KEY` in Secrets using `.streamlit/secrets.toml.example` as a template.

## Deployment Notes

- This repo is ready for Streamlit Community Cloud deployment.
- Main file: `app.py`
- Dependencies: `requirements.txt`
- Keep `.streamlit/secrets.toml` out of source control (already in `.gitignore`).

## Viva-Friendly Theory

### What is Cosine Similarity?
Cosine similarity measures the angle between two vectors.
- Value close to 1: very similar
- Value close to 0: dissimilar

It works well for text because it focuses on orientation of vectors rather than absolute magnitude.

### CountVectorizer vs TF-IDF
- CountVectorizer: uses term frequency counts
- TF-IDF: downweights common terms, highlights informative terms

## Resume Description

Developed a content-based recommendation system using Python and Scikit-learn that suggests similar movies based on cosine similarity. Implemented text vectorization techniques and built an interactive Streamlit UI for real-time recommendations.

## Future Enhancements

- Add collaborative filtering module
- Add hybrid recommender ranking
- Add user feedback loop for personalization
