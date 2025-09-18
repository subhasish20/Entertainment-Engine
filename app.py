import pickle
import pandas as pd
import streamlit as st

# Load saved DataFrame and cosine similarity matrix
with open('model/bert_embeddings.pkl', 'rb') as f:
    df = pickle.load(f)

with open('model/cosine_similarity.pkl', 'rb') as f:
    cosine_sim = pickle.load(f)

def recommend(movie_title, top_n=5):
    # Find the index of the selected movie (case-insensitive)
    idx_list = df.index[df['title'].str.lower() == movie_title.strip().lower()].tolist()
    if not idx_list:
        return []

    idx = idx_list[0]
    # Compute similarity scores and sort
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get indices of top similar movies (excluding the movie itself at position 0)
    rec_indices = [i for i, _ in sim_scores[1:top_n + 1]]
    return df.iloc[rec_indices]['title'].tolist()

# Streamlit UI
st.title('🎬 Movie Recommender')

movie_list = sorted(df['title'].dropna().unique().tolist())
selected_movie = st.selectbox("Select a movie", movie_list)

if st.button('Show Recommendations'):
    recs = recommend(selected_movie, top_n=5)
    if not recs:
        st.error("Movie not found in the database.")
    else:
        st.subheader('Recommended Movies:')
        for i, title in enumerate(recs, start=1):
            st.write(f"{i}. {title}")