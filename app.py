import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder

# =========================================================
# ⚙️ Load Model & Encoder
# =========================================================
@st.cache_resource
def load_model():
    try:
        model = pickle.load(open("train_model.pkl", "rb"))
        le = pickle.load(open("label_encoder.pkl", "rb"))
        return model, le
    except Exception as e:
        st.error("❌ Gagal memuat model atau encoder. Pastikan file 'model.pkl' dan 'label_encoder.pkl' ada di folder yang sama.")
        st.stop()

model, le = load_model()

# =========================================================
# 🧠 Judul & Deskripsi Aplikasi
# =========================================================
st.set_page_config(page_title="Mental Health Detector", layout="centered")
st.title("🧠 Mental Health Depression Detector")
st.markdown("""
Aplikasi ini menggunakan model *Machine Learning (Naive Bayes)* untuk memprediksi kemungkinan seseorang mengalami **depresi**  
berdasarkan data kesehatan mental.  

_Model ini dilatih menggunakan dataset nasional kesehatan mental._
""")

st.divider()

# =========================================================
# Input Data Responden
# =========================================================
st.subheader("Masukkan Data Responden")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Usia", min_value=10, max_value=100, value=25)
    gender = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
    occupation = st.selectbox("Pekerjaan", ["Pelajar", "Mahasiswa", "Karyawan", "Pengangguran", "Lainnya"])

with col2:
    st.markdown("### Penilaian Gejala Depresi (PHQ-9)")
    st.write("""
    Jawab berdasarkan kondisi Anda selama **2 minggu terakhir.**  
    PHQ-9 digunakan untuk menilai tingkat keparahan gejala depresi.
    """)
    
    phq_score = st.slider(
        "Seberapa sering Anda merasa sedih, kehilangan minat, atau sulit melakukan aktivitas?",
        min_value=0, max_value=27, value=10,
        help="0 = Tidak Pernah, 27 = Sangat Sering Mengalami Gejala Depresi"
    )

    # Interpretasi otomatis PHQ-9
    if phq_score <= 4:
        level = "Tidak ada / Minimal"
        color = "🟢"
    elif phq_score <= 9:
        level = "Depresi ringan"
        color = "🟡"
    elif phq_score <= 14:
        level = "Depresi sedang"
        color = "🟠"
    elif phq_score <= 19:
        level = "Depresi cukup berat"
        color = "🔴"
    else:
        level = "Depresi berat"
        color = "⚫"

    color_map = {
        "🟢": "#b7efc5",
        "🟡": "#fff6a5",
        "🟠": "#ffd6a5",
        "🔴": "#ffadad",
        "⚫": "#c0c0c0"
    }

    st.markdown(
        f"""
        <div style='background-color:{color_map[color]}; padding:10px; border-radius:10px; text-align:center;'>
            <strong>Skor PHQ-9 Anda: {phq_score}</strong><br>
            {color} <b>{level}</b>
        </div>
        """,
        unsafe_allow_html=True
    )

    sleep_hours = st.slider("Rata-rata Jam Tidur per Hari", 0, 12, 7)
    exercise_freq = st.slider("Frekuensi Olahraga per Minggu", 0, 7, 2)

st.divider()

# =========================================================
# Encoding Input Data
# =========================================================
gender_encoded = 1 if gender == "Perempuan" else 0

input_data = pd.DataFrame([{
    "age": age,
    "gender": gender_encoded,
    "occupation": occupation,
    "phq_score": phq_score,
    "sleep_hours": sleep_hours,
    "exercise_freq": exercise_freq
}])

input_data = pd.get_dummies(input_data)

# =========================================================
# Prediksi Depresi
# =========================================================
st.subheader("Hasil Prediksi")

if st.button("Deteksi Tingkat Depresi"):
    try:
        prediction = model.predict(input_data)[0]
        result_label = le.inverse_transform([prediction])[0]
        
        if "depresi" in result_label.lower():
            st.error(f"Hasil Deteksi: **{result_label.upper()}** 😔")
            st.markdown("> Disarankan untuk berkonsultasi dengan profesional kesehatan mental.")
        else:
            st.success(f"Hasil Deteksi: **{result_label.upper()}** 😄")
            st.markdown("> Tetap jaga kesehatan mental dan fisikmu!")
    except Exception as e:
        st.error(f"Terjadi kesalahan dalam proses prediksi: {e}")

st.divider()

# =========================================================
# Informasi Tambahan
# =========================================================
st.subheader(" Tentang Aplikasi")
st.markdown("""
- Algoritma utama: **Naive Bayes (GaussianNB)**
- Data latih: Dataset nasional kesehatan mental (versi praproses)
- Fitur utama: PHQ-9 score, usia, jam tidur, kebiasaan olahraga, dan jenis kelamin
- Tujuan: Edukasi dan penelitian untuk kesadaran kesehatan mental
""")

st.caption("© 2025 Mental Health Detection App — powered by Streamlit & scikit-learn")
