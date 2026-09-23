import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

st.set_page_config(page_title="Rekomendasi Film Netflix", page_icon="🎬")

st.title("🎬 Sistem Rekomendasi Film Netflix")
st.write(
    "Cari sebuah judul film, dan sistem akan merekomendasikan film serupa "
    "berdasarkan director, cast, country, rating, genre, dan deskripsi."
)


@st.cache_data
def load_data():
    df = pd.read_csv("netflix_titles.csv")
    df['director'] = df['director'].fillna('unknown')
    df['cast'] = df['cast'].fillna('unknown')
    df['country'] = df['country'].fillna('unknown')
    df = df.dropna(subset=['date_added', 'duration', 'rating']).reset_index(drop=True)

    soup_cols = ['director', 'cast', 'country', 'rating', 'listed_in', 'description']
    df['soup'] = df[soup_cols].astype(str).agg(' '.join, axis=1)
    return df


@st.cache_resource
def build_model(_df):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(_df['soup'])
    model_nn = NearestNeighbors(n_neighbors=6, metric='cosine', algorithm='brute')
    model_nn.fit(tfidf_matrix)
    return tfidf_matrix, model_nn


def get_recommendations(df, tfidf_matrix, model_nn, title_input, n_rekomendasi=5):
    try:
        idx = df[df['title'].str.lower() == title_input.lower()].index[0]
    except IndexError:
        return None

    query_vector = tfidf_matrix[idx]
    distances, indices = model_nn.kneighbors(query_vector, n_neighbors=len(df))
    distances = distances.flatten()
    indices = indices.flatten()

    recom_list = []
    for i in range(len(indices)):
        if indices[i] == idx:
            continue
        sim_score = 1 - distances[i]
        if sim_score > 0:
            recom_list.append({
                'title': df.iloc[indices[i]]['title'],
                'similarity': round(sim_score, 3)
            })
        if len(recom_list) == n_rekomendasi:
            break

    return pd.DataFrame(recom_list)


def calculate_genre_overlap(target_genres, rec_genres):
    if not target_genres or not rec_genres:
        return 0.0
    target_set = set(target_genres)
    rec_set = set(rec_genres)
    overlap = target_set.intersection(rec_set)
    return len(overlap) / len(target_set)


def evaluate_recommender_system(df, tfidf_matrix, model_nn, sample_size=100):
    sample_df = df.sample(n=sample_size, random_state=42) if len(df) > sample_size else df
    all_scores = []

    for _, row in sample_df.iterrows():
        target_title = row["title"]
        target_genres = row["listed_in"].split(",") if isinstance(row["listed_in"], str) else []

        recommendations = get_recommendations(df, tfidf_matrix, model_nn, target_title, n_rekomendasi=5)
        if recommendations is None or recommendations.empty:
            continue

        movie_scores = []
        for rec_title in recommendations['title']:
            rec_row = df[df["title"] == rec_title].iloc[0]
            rec_genres = rec_row["listed_in"].split(",") if isinstance(rec_row["listed_in"], str) else []
            movie_scores.append(calculate_genre_overlap(target_genres, rec_genres))

        if movie_scores:
            all_scores.append(np.mean(movie_scores))

    return np.mean(all_scores) if all_scores else 0.0


# ---------- Load data & model (di-cache, hanya jalan sekali) ----------
df = load_data()
tfidf_matrix, model_nn = build_model(df)

# ---------- UI pencarian ----------
st.subheader("Cari Rekomendasi")

col1, col2 = st.columns([3, 1])
with col1:
    title_input = st.text_input("Masukkan judul film (contoh: Zodiac, The Conjuring)")
with col2:
    n_rekomendasi = st.number_input("Jumlah rekomendasi", min_value=1, max_value=20, value=5)

if st.button("Cari Rekomendasi", type="primary"):
    if not title_input.strip():
        st.warning("Masukkan judul film terlebih dahulu.")
    else:
        hasil = get_recommendations(df, tfidf_matrix, model_nn, title_input, n_rekomendasi=n_rekomendasi)
        if hasil is None:
            st.error(f"Maaf, judul '{title_input}' tidak ditemukan dalam database.")
        else:
            st.success(f"Rekomendasi untuk '{title_input}':")
            st.dataframe(hasil, use_container_width=True)

st.divider()

# ---------- Eksplorasi data (opsional) ----------
with st.expander("Lihat data mentah"):
    st.dataframe(df.head(50), use_container_width=True)
    st.caption(f"Total baris setelah pembersihan: {len(df)}")

# ---------- Evaluasi sistem (opsional, agak berat) ----------
with st.expander("Evaluasi Kualitas Rekomendasi"):
    st.write(
        "Menghitung rata-rata genre overlap dari 100 sampel film acak, "
        "untuk mengukur seberapa relevan rekomendasi sistem ini secara keseluruhan."
    )
    if st.button("Jalankan Evaluasi"):
        with st.spinner("Menghitung skor evaluasi..."):
            avg_score = evaluate_recommender_system(df, tfidf_matrix, model_nn, sample_size=100)
        st.metric("Rata-rata Genre Overlap", f"{avg_score:.3f}")