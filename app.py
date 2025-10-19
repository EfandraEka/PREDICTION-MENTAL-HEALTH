import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder

# =========================================================
# Load Model & Encoder
# =========================================================
@st.cache_resource
def load_model():
    try:
        model = pickle.load(open("model.pkl", "rb"))
        le = pickle.load(open("label_encoder.pkl", "rb"))
        return model, le
    except Exception as e:
        st.error(" Gagal memuat model atau encoder. Pastikan file 'model.pkl' dan 'label_encoder.pkl' ada di folder yang sama.")
        st.stop()

model, le = load_model()

# =========================================================
# Judul & Deskripsi
# =========================================================
st.set_page_config(page_title="Mental Health Detector", layout="centered")
st.title(" Mental Health Depression Detector")
st.markdown("""
Aplikasi ini menggunakan model *Machine Learning (Naive Bayes)* untuk memprediksi kemungkinan seseorang mengalami **depresi**  
berdasarkan data kesehatan mental.  

_Model ini dilatih menggunakan dataset nasional kesehatan mental._
""")

st.divider()

# =========================================================
# Input User
# =========================================================
st.subheader(" Masukkan Data Responden")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Usia", min_value=10, max_value=100, value=25)
    gender = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
    occupation = st.selectbox("Pekerjaan", ["Pelajar", "Mahasiswa", "Karyawan", "Pengangguran", "Lainnya"])

with col2:
    phq_score = st.slider("Skor PHQ-9 (0-27)", 0, 27, 10)
    sleep_hours = st.slider("Rata-rata Jam Tidur per Hari", 0, 12, 7)
    exercise_freq = st.slider("Frekuensi Olahraga per Minggu", 0, 7, 2)

st.divider()

# =========================================================
# Encoding Input
# =========================================================
# Contoh encoding sederhana
gender_encoded = 1 if gender == "Perempuan" else 0

# Bisa disesuaikan tergantung fitur dataset kamu
input_data = pd.DataFrame([{
    "age": age,
    "gender": gender_encoded,
    "occupation": occupation,
    "phq_score": phq_score,
    "sleep_hours": sleep_hours,
    "exercise_freq": exercise_freq
}])

# One-hot encoding jika ada fitur kategorikal
input_data = pd.get_dummies(input_data)

# Pastikan kolom sama seperti data training
# (opsional: sesuaikan dengan X_train.columns)
# Contoh:
# input_data = input_data.reindex(columns=model_columns, fill_value=0)

# =========================================================
# 🧩 Prediksi
# =========================================================
if st.button("Deteksi Depresi"):
    try:
        prediction = model.predict(input_data)[0]
        result_label = le.inverse_transform([prediction])[0]
        
        if "depresi" in result_label.lower():
            st.error(f" Hasil Deteksi: **{result_label.upper()}** ")
            st.markdown("> Disarankan untuk berkonsultasi dengan profesional kesehatan mental.")
        else:
            st.success(f" Hasil Deteksi: **{result_label.upper()}** ")
            st.markdown("> Tetap jaga kesehatan mental dan fisikmu!")
    except Exception as e:
        st.error(f"Terjadi kesalahan dalam proses prediksi: {e}")

st.divider()

# =========================================================
#  Info Tambahan
# =========================================================
st.subheader("Tentang Aplikasi")
st.markdown("""
- Algoritma utama: **Naive Bayes (GaussianNB)**
- Data latih: Dataset nasional kesehatan mental (versi praproses)
- Fitur utama: PHQ-9 score, usia, jam tidur, dan kebiasaan olahraga
- Dikembangkan untuk penelitian dan edukasi
""")

st.caption("© 2025 Mental Health Detection App — powered by Streamlit & scikit-learn")
