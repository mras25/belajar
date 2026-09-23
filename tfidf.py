# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# %%
df = pd.read_csv(r"E:\belajar\archive (15)\netflix_titles.csv")
df

# %% [markdown]
# terdapat 8807 baris, terlihat cukup banyak nan, durasi movie menggunakan menit, durasi tv show menggunakan season

# %%
df.isna().sum()

# %% [markdown]
# nan pada director terlalu banyak, klo didropna hampir 1/3 total baris hilang

# %%
df.duplicated().sum()

# %%
df['director'] = df['director'].fillna('unknown')
df['cast'] = df['cast'].fillna('unknown')
df['country'] = df['country'].fillna('unknown')
df = df.dropna(subset = ['date_added', 'duration', 'rating']).reset_index(drop=True)

# %% [markdown]
# director, cast dan country diisi unknown karena terlalu banyak jika dihapus, lalu untuk sisanya hanya karena sedikit di drop saja

# %%
#def remove_spasi(text):
#    return [i.replace(" ","") for i in text]

#df = df.apply(remove_spasi, include = 'object')

# %% [markdown]
# errornya akan seperti ini karena Kode tersebut tidak bisa berjalan langsung jika variabel include digunakan di dalam metode .apply() milik pandas DataFrame.Penyebab utamanya adalah metode df.apply() tidak memiliki parameter bernama include. Parameter include biasanya digunakan pada metode df.select_dtypes(include='object'). Selain itu, jika Anda ingin menerapkan fungsi ke setiap elemen atau kolom/baris, ada cara yang lebih tepat di pandas.
# 
# ---------------------------------------------------------------------------
# TypeError                                 Traceback (most recent call last)
# Cell In[18], line 4
#       1 def remove_spasi(text):
#       2     return [i.replace(" ","") for i in text]
#       3 
# ----> 4 df = df.apply(remove_spasi, include = 'object')
# 
# File c:\Users\syahr\.conda\envs\kerja\Lib\site-packages\pandas\core\frame.py:12437, in DataFrame.apply(self, func, axis, raw, result_type, args, by_row, engine, engine_kwargs, **kwargs)
#   12433                 engine_kwargs=engine_kwargs,
#   12434                 args=args,
#   12435                 kwargs=kwargs,
#   12436             )
# > 12437             return op.apply().__finalize__(self, method="apply")
#   12438         elif hasattr(engine, "__pandas_udf__"):
#   12439             if result_type is not None:
#   12440                 raise NotImplementedError(
# 
# File c:\Users\syahr\.conda\envs\kerja\Lib\site-packages\pandas\core\apply.py:1015, in FrameApply.apply(self)
#    1012 elif self.raw:
#    1013     return self.apply_raw(engine=self.engine, engine_kwargs=self.engine_kwargs)
# -> 1015 return self.apply_standard()
# 
# File c:\Users\syahr\.conda\envs\kerja\Lib\site-packages\pandas\core\apply.py:1167, in FrameApply.apply_standard(self)
#    1165 def apply_standard(self):
# ...
#    1185         # If we have a view on v, we need to make a copy because
#    1186         #  series_generator will swap out the underlying data
#    1187         results[i] = results[i].copy(deep=False)
# 
# TypeError: remove_spasi() got an unexpected keyword argument 'include'
# Output is truncated. View as a scrollable element or open in a text editor. Adjust cell output settings...

# %%
#kolom_object = df.select_dtypes(include='object').columns
kolom_soup = ['director', 'cast', 'country', 'listed_in', 'description']
for col in kolom_soup:
    df[col] = df[col].astype(str).str.replace(" ", "")


# %% [markdown]
# select_dtypes(include='object') mengambil SEMUA kolom bertipe object — bukan cuma director, cast, country, listed_in yang memang ingin dibersihkan untuk soup, tapi juga ikut kena: show_id, type, title, date_added, description.
# 
# Akibatnya kolom title ikut kehilangan spasinya:
# 
# "Kota Factory" → "KotaFactory"
# "The Conjuring" → "TheConjuring"
# 
# Sementara di bagian pencarian:
# 
# python
# title_tes_1 = "Kota Factory"   # masih pakai spasi
# idx = df[df['title'].str.lower() == title_input.lower()].index[0]
# 
# title_input.lower() = "kota factory" (ada spasi) dicocokkan dengan isi kolom title yang sudah jadi "kotafactory" (tanpa spasi) → tidak pernah match → IndexError → tertangkap di except → muncul pesan "tidak ditemukan"

# %%
#df['soup'] = ( df['director'], df['cast'], df['country'], df['release_year'], df['rating'], df['duration'], df['listed_in'])
#df['soup'] = df['soup'].apply(lambda x: ' '.join(x))

# %% [markdown]
# 

# %% [markdown]
# ini terjadi karena mencoba memasukkan sebuah tuple yang berisi 7 kolom ke dalam satu kolom baru (df['soup'])
# 
# ---------------------------------------------------------------------------
# ValueError                                Traceback (most recent call last)
# Cell In[23], line 1
# ----> 1 df['soup'] = ( df['director'], df['cast'], df['country'], df['release_year'], df['rating'], df['duration'], df['listed_in'])
#       2 df['soup'] = df['soup'].apply(lambda x: ' '.join(x))
# 
# File c:\Users\syahr\.conda\envs\kerja\Lib\site-packages\pandas\core\frame.py:4672, in DataFrame.__setitem__(self, key, value)
#    4668             # Column to set is duplicated
#    4669             self._setitem_array([key], value)
#    4670         else:
#    4671             # set column
# -> 4672             self._set_item(key, value)
# 
# File c:\Users\syahr\.conda\envs\kerja\Lib\site-packages\pandas\core\frame.py:4874, in DataFrame._set_item(self, key, value)
#    4870 
#    4871         Series/TimeSeries will be conformed to the DataFrames index to
#    4872         ensure homogeneity.
#    4873         """
# -> 4874         value, refs = self._sanitize_column(value)
#    4875 
#    4876         if (
#    4877             key in self.columns
# 
# File c:\Users\syahr\.conda\envs\kerja\Lib\site-packages\pandas\core\frame.py:5756, in DataFrame._sanitize_column(self, value)
#    5752                 value = Series(value)
#    5753             return _reindex_for_setitem(value, self.index)
#    5754 
#    5755         if is_list_like(value):
# -> 5756             com.require_length_match(value, self.index)
#    5757         return sanitize_array(value, self.index, copy=True, allow_2d=True), None
# 
# File c:\Users\syahr\.conda\envs\kerja\Lib\site-packages\pandas\core\common.py:601, in require_length_match(data, index)
#     597 """
#     598 Check the length of data matches the length of the index.
#     599 """
#     600 if len(data) != len(index):
# --> 601     raise ValueError(
#     602         "Length of values "
#     603         f"({len(data)}) "
#     604         "does not match length of index "
#     605         f"({len(index)})"
#     606     )
# 
# ValueError: Length of values (7) does not match length of index (8790)

# %%
# Menggabungkan kolom dengan spasi sebagai pemisah
soup = ['director', 'cast', 'country', 'rating', 'listed_in', 'description']

df['soup'] = df[soup].astype(str).agg(' '.join, axis=1)

df['soup']


# %%
titles = [f"title film ke - {i}" for i in range (8806)]
query_title = ['title film ke-200']

# %%
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['soup'])   # hanya kolom soup, bukan df utuh
fitur_names = vectorizer.get_feature_names_out()
df_tfidf = pd.DataFrame(tfidf_matrix.toarray(), columns=fitur_names)
query_tfidf = vectorizer.transform(query_title)

# %% [markdown]
# Tahap 1: Vektorisasi teks
# 
# Pertanyaan: "Teks (director, cast, genre) harus jadi angka. Angka yang adil itu seperti apa?"
# 
# python
# soup = ['director', 'cast', 'country', 'rating', 'listed_in', 'description']
# df['soup'] = df[soup].astype(str).agg(' '.join, axis=1)
# 
# Ini jawaban untuk "gimana caranya satu film direpresentasikan sebagai satu kesatuan teks?" — kamu gabung semua fitur relevan jadi satu string panjang per baris. Ini masih tahap persiapan, belum menjawab "angka yang adil" — makanya baris berikutnya baru masuk ke inti masalahnya:
# 
# python
# vectorizer = TfidfVectorizer()
# tfidf_matrix = vectorizer.fit_transform(df['soup'])
# 
# Di sinilah pertanyaan "angka yang adil" dijawab. TfidfVectorizer() tidak sekadar menghitung berapa kali sebuah kata muncul (itu baru CountVectorizer) — dia juga menghukum kata yang terlalu sering muncul di banyak dokumen (seperti "unknown" yang muncul di ribuan baris karena director NaN) dan memberi bobot lebih ke kata yang jarang tapi khas (seperti "JulienLeclercq" yang cuma muncul di 1-2 film). fit_transform() melakukan 2 hal sekaligus: mempelajari kosakata dari seluruh 8790 film (fit), lalu mengubah tiap baris jadi vektor angka berdasarkan kosakata itu (transform). Hasilnya tfidf_matrix — tiap baris = 1 film, tiap kolom = 1 kata unik, isinya bobot TF-IDF.

# %% [markdown]
# TfidfVectorizer(): Membuat objek pencari bobot teks. Secara otomatis, fungsi ini juga melakukan tokenization (memecah kalimat jadi kata) dan menghapus tanda baca.
# fit_transform(): Proses mempelajari kosakata dari seluruh dokumen sekaligus menghitung nilai TF-IDF untuk setiap kata di setiap kalimat.
# get_feature_names_out(): Mengambil daftar semua kata unik yang berhasil dideteksi dalam teks.
# toarray(): Mengubah bentuk matriks bawaan scikit-learn (sparse matrix) menjadi array biasa agar nilainya bisa kita masukkan ke tabel

# %%
model_nn = NearestNeighbors(n_neighbors=6, metric= 'cosine', algorithm='brute')
model_nn.fit(tfidf_matrix)

# %% [markdown]
# Tahap 2: Hitung kemiripan
# 
# Pertanyaan: "Sekarang tiap film = deretan angka. 'Mirip' itu diukur gimana? Dan kalau ada 8790 film, apa efisien membandingkan semua ke semua?"
# 
# python
# model_nn = NearestNeighbors(n_neighbors=6, metric='cosine', algorithm='brute')
# model_nn.fit(tfidf_matrix)
# 
# metric='cosine' menjawab pertanyaan pertama — ini rumus untuk mengukur "seberapa searah" dua vektor (bukan jarak lurus, tapi sudut antar vektor; dua film dianggap mirip kalau arah vektornya mirip, terlepas dari "panjang" vektornya).
# 
# NearestNeighbors (bukan cosine_similarity penuh) menjawab pertanyaan kedua. Bedanya krusial: cosine_similarity(tfidf_matrix, tfidf_matrix) akan langsung menghitung matriks 8790×8790 di awal — semua pasangan dihitung sekaligus, walau kamu cuma butuh 3-5 tetangga terdekat per film. NearestNeighbors.fit() cuma menyimpan struktur data, belum menghitung apa-apa — perhitungan jarak baru terjadi nanti, hanya untuk film yang benar-benar ditanya (n_neighbors=6 artinya nanti dia akan cari 6 tetangga terdekat saat diminta, bukan menghitung ke semua 8789 film lain).

# %%
def get_recommendations(title_input, n_rekomendasi=3):
    try:
        idx = df[df['title'].str.lower() == title_input.lower()].index[0]
    except IndexError:
        return f"Maaf, title '{title_input}' tidak ditemukan dalam database."
    
    
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
                'similarity': sim_score
            })
            
        if len(recom_list) == n_rekomendasi:
            break
            
    return pd.DataFrame(recom_list)

# %% [markdown]
# Tahap 3: Bangun fungsi rekomendasi
# 
# Pertanyaan: "User ngetik JUDUL, bukan angka. Gimana menghubungkan dunia manusia ke dunia vektor?"
# 
# python
# def get_recommendations(title_input, n_rekomendasi=3):
#     try:
#         idx = df[df['title'].str.lower() == title_input.lower()].index[0]
#     except IndexError:
#         return f"Maaf, title '{title_input}' tidak ditemukan dalam database."
# 
# Ini langkah pertama jembatan: judul → index. User kasih string, kamu cari posisinya di dataframe. try/except di sini menjawab pertanyaan turunan: "gimana kalau judulnya salah ketik / tidak ada?" — daripada program crash, kasih pesan yang jelas.
# 
# python
#     query_vector = tfidf_matrix[idx]
#     distances, indices = model_nn.kneighbors(query_vector, n_neighbors=len(df))
# 
# Ini jembatan kedua: index → vektor → cari tetangga. tfidf_matrix[idx] mengambil "representasi angka" dari film yang dicari (hasil kerja Tahap 1). model_nn.kneighbors(...) memakai model dari Tahap 2 untuk mencari film-film dengan vektor paling mirip.
# 
# python
#     recom_list = []
#     for i in range(len(indices)):
#         if indices[i] == idx:
#             continue  # Lewati jika itu dirinya sendiri
#         sim_score = 1 - distances[i]
#         if sim_score > 0:
#             recom_list.append({'title': df.iloc[indices[i]]['title'], 'similarity': sim_score})
#         if len(recom_list) == n_rekomendasi:
#             break
#     return pd.DataFrame(recom_list)
# 
# Ini jembatan terakhir: index hasil pencarian → judul lagi, supaya manusia bisa baca (df.iloc[indices[i]]['title']). if indices[i] == idx: continue menjawab pertanyaan turunan lain: "bukankah film itu pasti paling mirip sama dirinya sendiri? Itu bukan rekomendasi yang berguna" — jadi disingkirkan.

# %%
print("--- UJI COBA 1: ")
title_tes_1 = "zodiac"
print(f"Rekomendasi untuk: '{title_tes_1}'")
print(get_recommendations(title_tes_1, n_rekomendasi=3))

print("\n--- UJI COBA 2:")
title_tes_2 = "The Conjuring"
print(f"Rekomendasi untuk: '{title_tes_2}'")
print(get_recommendations(title_tes_2, n_rekomendasi=2))

# %%
def calculate_genre_overlap(target_genres, rec_genres):

    if not target_genres or not rec_genres:
        return 0.0

    target_set = set(target_genres)
    rec_set = set(rec_genres)

    overlap = target_set.intersection(rec_set)

    return len(overlap) / len(target_set)


def evaluate_recommender_system(df, recommend_function, sample_size=100):
    if len(df) > sample_size:
        sample_df = df.sample(n=sample_size, random_state=42)
    else:
        sample_df = df

    all_scores = []

    for idx, row in sample_df.iterrows():
        target_title = row["title"]
        target_genres = (
            row["listed_in"].split(",")         
            if isinstance(row["listed_in"], str)
            else []
        )

        try:
            recommendations = recommend_function(target_title, n_rekomendasi=5)
            if isinstance(recommendations, str):
                continue

            movie_scores = []                    

            for rec_title in recommendations['title']:
                rec_row = df[df["title"] == rec_title].iloc[0]
                rec_genres = rec_row["listed_in"].split(",") if isinstance(rec_row["listed_in"], str) else []
                score = calculate_genre_overlap(target_genres, rec_genres)
                movie_scores.append(score)       

            if movie_scores:
                all_scores.append(np.mean(movie_scores))

        except Exception as e:
            print(f"Error di '{target_title}': {e}")  
            continue

    macro_average_overlap = np.mean(all_scores) if all_scores else 0.0
    return macro_average_overlap

# %% [markdown]
# Tahap 4: Evaluasi kualitatif
# 
# Pertanyaan: "Rekomendasinya kelihatan masuk akal pas dicoba manual — tapi itu 'beneran bagus' atau kebetulan?"
# 
# python
# def calculate_genre_overlap(target_genres, rec_genres):
#     target_set = set(target_genres)
#     rec_set = set(rec_genres)
#     overlap = target_set.intersection(rec_set)
#     return len(overlap) / len(target_set)
# 
# Ini jawaban atas "aku butuh angka, bukan feeling". Karena tidak ada "jawaban benar" resmi untuk rekomendasi film, kamu pilih proksi yang masuk akal: kalau sistem benar-benar menangkap kemiripan, genre-nya harusnya nyambung. Fungsi ini menghitung berapa persen genre film target yang juga ada di film rekomendasi.
# 
# python
# def evaluate_recommender_system(df, recommend_function, sample_size=100):
#     sample_df = df.sample(n=sample_size, random_state=42)
#     all_scores = []
#     for idx, row in sample_df.iterrows():
#         recommendations = recommend_function(row['title'], n_rekomendasi=5)
#         ...
#         all_scores.append(np.mean(movie_scores))
#     return np.mean(all_scores)
# 
# Ini jawaban atas pertanyaan turunan: "satu contoh manual gak cukup buat menyimpulkan 'bagus' — gimana caranya uji banyak film sekaligus tanpa harus cek satu-satu dengan mata?" Jawabannya: sampling otomatis — ambil 100 film acak (random_state=42 supaya hasilnya bisa diulang, tidak berubah-ubah tiap dijalankan), jalankan fungsi rekomendasi ke semuanya, rata-ratakan skornya. Hasil akhirnya 0.642 adalah rangkuman objektif dari 100 kali percobaan, bukan cuma 2 contoh yang kamu pilih manual (Zodiac, The Conjuring).

# %%
avg_score = evaluate_recommender_system(df, get_recommendations, sample_size=100)
print(f"Rata-rata genre overlap: {avg_score:.3f}")


