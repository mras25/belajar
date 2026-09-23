import streamlit as st
import pandas as pd
import pickle

st.title("Rekomendasi Film Netflix")

@st.cache_resource
def load_model():
    with open("model_artifacts.pkl", "rb") as f:
        artifacts = pickle.load(f)
    return artifacts['df'], artifacts['tfidf_matrix'], artifacts['model_nn']

df, tfidf_matrix, model_nn = load_model()

def get_recommendations(title_input, n_rekomendasi=5):
    try:
        idx = df[df['title'].str.lower() == title_input.lower()].index[0]
    except IndexError:
        return None

    distances, indices = model_nn.kneighbors(tfidf_matrix[idx], n_neighbors=len(df))
    distances, indices = distances.flatten(), indices.flatten()

    recom_list = []
    for i in range(len(indices)):
        if indices[i] == idx:
            continue
        sim_score = 1 - distances[i]
        if sim_score > 0:
            recom_list.append({
                'title': df.iloc[indices[i]]['title'],
                'genre': df.iloc[indices[i]]['listed_in'],
                'similarity': round(sim_score, 3)
            })
        if len(recom_list) == n_rekomendasi:
            break
    return pd.DataFrame(recom_list)

title_input = st.text_input("Masukkan judul film:")
if st.button("Cari rekomendasi"):
    hasil = get_recommendations(title_input)
    if hasil is None:
        st.write(f"Judul '{title_input}' tidak ditemukan.")
    else:
        st.write(hasil)