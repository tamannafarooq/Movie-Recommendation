# ML PROJECT

## Content-Based Recommendation System using Cosine Similarity

Submitted by: Farha  
Program: B.Tech (CSE)  
Session: Summer Internship 2026

Deployed Application: https://movie-recommendation-bqfvhbuotejgna7shxbmx3.streamlit.app/

GitHub Repository: https://github.com/Farha-n/Movie-recommendation

---

## Certificate Page (Template)

This is to certify that the project report titled **"Content-Based Recommendation System using Cosine Similarity"** is the bonafide work carried out by **Farha** during the ML project period under the guidance of the project mentor and departmental supervision.

Supervisor Signature: ____________________  
HOD Signature: ____________________  
Date: ____________________

---

## Self Declaration

I hereby declare that the work presented in this report titled **"Content-Based Recommendation System using Cosine Similarity"** is my original work carried out during ML project training. This report has not been submitted elsewhere for any other academic award.

Name: Tamana Farooq Khanday
Signature: ____________________  
Date: ____________________

---

## Acknowledgement

I would like to express my sincere gratitude to my faculty mentors and internship guides for their valuable support during this ML project. Their guidance in machine learning concepts, project structuring, and deployment practices helped me complete this work successfully.

I also thank my peers and family for constant encouragement and feedback during development and testing.

---

## Abstract

Recommendation systems are widely used in digital platforms to personalize user experience and improve content discovery. This project presents a **Content-Based Movie Recommendation System** that suggests similar movies based on textual attributes such as genres, keywords, and overview.

The system preprocesses movie metadata, converts text into vector representations using **CountVectorizer** or **TF-IDF**, and computes pairwise similarity using **Cosine Similarity**. Based on the selected movie, the application returns top-N relevant recommendations.

A complete interactive web application was developed using **Streamlit**, including recommendation cards, optional poster integration using TMDB API, and an evaluation dashboard with proxy metrics (Genre Match@K, Catalog Coverage, Mean Similarity). The project is deployed online and accessible for real-time demonstration.

---

## Table of Contents

1. Introduction  
2. Problem Statement and Objectives  
3. Literature Survey  
4. System Design and Methodology  
5. Implementation Details  
6. Results and Evaluation  
7. Deployment and User Manual  
8. Conclusion and Future Scope  
9. References  
10. Appendix

---

## 1. Introduction

### 1.1 Background

Modern users are exposed to massive content catalogs. Finding relevant items manually is difficult and time-consuming. Recommendation systems solve this by automatically ranking items based on user context or item similarity.

### 1.2 Need for the Project

In many academic mini-projects, recommendation systems are explained theoretically but not deployed as usable products. This project focuses on both machine learning logic and practical web deployment.

### 1.3 Domain

- Machine Learning
- Natural Language Processing (basic text feature engineering)
- Recommender Systems
- Web App Development

---

## 2. Problem Statement and Objectives

### 2.1 Problem Statement

Given a movie title, recommend similar movies based on their content features.

### 2.2 Objectives

- Build an end-to-end content-based recommender system.
- Use robust text preprocessing and vectorization.
- Implement cosine similarity ranking logic.
- Provide an interactive Streamlit interface.
- Add quality-oriented features such as evaluation metrics and artifact caching.
- Deploy the project for live demonstration.

### 2.3 Scope

- Focused on movie recommendation using metadata.
- Does not use user watch history (non-collaborative model).
- Works with custom CSV and can be extended to larger TMDB datasets.

---

## 3. Literature Survey

### 3.1 Overview of Recommendation Approaches

- Content-Based Filtering: recommends items similar to selected item attributes.
- Collaborative Filtering: uses behavior patterns of similar users.
- Hybrid Systems: combines both to improve accuracy and robustness.

### 3.2 Why Content-Based for This Project

- Easier to explain in viva and implement from scratch.
- Does not require user-item interaction matrix.
- Works well with textual metadata fields.

### 3.3 Related Concepts

- Text preprocessing (cleaning, stop words, token features)
- Vectorization (CountVectorizer, TF-IDF)
- Similarity functions (Cosine Similarity)

---

## 4. System Design and Methodology

### 4.1 Dataset and Features

Input columns used:
- title
- genres
- keywords
- overview

Feature engineering:
- Merge genres + keywords + overview into a single text field called `tags`
- Clean and normalize text (lowercasing, punctuation removal, whitespace normalization)

### 4.2 Vectorization

Two selectable methods in UI:
- CountVectorizer
- TF-IDF Vectorizer

### 4.3 Similarity Computation

Cosine similarity formula:

$$
\text{cosine}(A, B) = \frac{A \cdot B}{\|A\|\|B\|}
$$

Where $A$ and $B$ are text feature vectors of two movies.

### 4.4 Recommendation Logic

1. Find index of selected movie.
2. Fetch similarity scores against all movies.
3. Sort scores in descending order.
4. Return top-N similar titles excluding selected title.

### 4.5 Advanced Components Implemented

- Local model artifact caching (`artifacts/`) for faster repeated runs.
- Optional TMDB poster retrieval by API key.
- Multi-tab app with Recommend, Evaluate, Explore Data, Viva Notes.

---

## 5. Implementation Details

### 5.1 Technology Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Requests

### 5.2 Project Structure

- app.py
- movies.csv
- requirements.txt
- README.md
- .streamlit/config.toml
- .streamlit/secrets.toml.example
- artifacts/ (auto-generated cache)

### 5.3 Key Modules

- Data loading and validation
- Feature preprocessing
- Model build/load with cache
- Recommendation generation
- Poster API integration
- Evaluation metrics display

---

## 6. Results and Evaluation

### 6.1 Functional Results

- User can select a movie from dropdown and get top recommendations.
- Similarity score is shown for each recommendation.
- Poster cards displayed when TMDB key is provided.

### 6.2 Evaluation Dashboard

Implemented proxy quality metrics:
- Genre Match@K
- Catalog Coverage
- Mean Similarity

These metrics help explain model behavior and quality in viva presentations.

### 6.3 Observations

- TF-IDF generally gives cleaner recommendations than raw counts on text-heavy fields.
- Increasing max_features can improve granularity but increases computation time.
- Caching significantly reduces repeated startup computation.

---

## 7. Deployment and User Manual

### 7.1 Live Deployment

App URL: https://movie-recommendation-bqfvhbuotejgna7shxbmx3.streamlit.app/

### 7.2 Steps to Run Locally

1. Install dependencies:
   - `pip install -r requirements.txt`
2. Start app:
   - `streamlit run app.py`
3. Open local URL shown in terminal.

### 7.3 Optional Poster Setup

1. Create TMDB API key.
2. Set environment variable `TMDB_API_KEY`.
3. Enable poster option in sidebar.

### 7.4 User Flow

1. Open app.
2. Choose vectorizer and settings.
3. Select movie.
4. Click Recommend.
5. View recommendations and quality metrics.

---

## 8. Conclusion and Future Scope

### 8.1 Conclusion

The project successfully demonstrates an end-to-end content-based recommender system with a production-style interface and live deployment. It combines ML fundamentals with practical software engineering and provides clear explainability for academic evaluation.

### 8.2 Future Scope

- Add collaborative filtering model and hybrid ranking.
- Integrate user feedback for personalization.
- Expand dataset to full TMDB metadata.
- Add A/B comparison dashboard for model variants.

---

## 9. References

1. Scikit-learn Documentation: https://scikit-learn.org/stable/  
2. Streamlit Documentation: https://docs.streamlit.io/  
3. TMDB API Docs: https://developer.themoviedb.org/docs  
4. Recommender Systems Overview (research references and survey papers)

---

## 10. Appendix

### Appendix A: Screenshots to Attach in Final Submission

- Home tab (Recommend)
- Evaluation dashboard
- Dataset explorer
- Deployed app page
- GitHub repository page

### Appendix B: Viva Questions (Quick Answers)

1. Why cosine similarity?  
   - It measures angle-based similarity and works well for sparse text vectors.

2. What is vectorization?  
   - Converting text into numeric features that ML algorithms can process.

3. Difference between CountVectorizer and TF-IDF?  
   - TF-IDF downweights common words and emphasizes informative terms.

4. Why content-based instead of collaborative filtering?  
   - Content-based does not require historical user behavior and is simpler for cold-start demo projects.

---

## Resume Description (Ready to Use)

Developed and deployed a content-based movie recommendation system using Python, Scikit-learn, and Streamlit. Implemented text preprocessing, TF-IDF/Count vectorization, cosine similarity ranking, local artifact caching, and an evaluation dashboard for model quality analysis.
