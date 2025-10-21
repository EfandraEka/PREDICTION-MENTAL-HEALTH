import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder

# =========================================================
# Load Model, Encoder, dan Kolom
# =========================================================
@st.cache_resource
def load_model():
    try:
        model = pickle.load(open("model.pkl", "rb"))
        le = pickle.load(open("label_encoder.pkl", "rb"))
        model_columns = pickle.load(open("model_columns.pkl", "rb"))  # === FIXED ===
        return model, le, model_columns
    except Exception as e:
        st.error("❌ Gagal memuat model atau encoder. Pastikan file 'model.pkl', 'label_encoder.pkl', dan 'model_columns.pkl' ada di folder yang sama.")
        st.stop()

model, le, model_columns = load_model()

# =========================================================
# Tampilan Aplikasi
# =========================================================
st.set_page_config(page_title="Mental Health Detector", layout="centered")
st.title("🧠 Mental Health Depression Detector")
st.markdown("""
Aplikasi ini menggunakan *Machine Learning (Naive Bayes)* untuk memprediksi kemungkinan seseorang mengalami **depresi**  
berdasarkan data kesehatan mental.
""")

st.divider()
st.subheader("🧍‍♀️ Masukkan Data Responden")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Usia", min_value=10, max_value=100, value=25)
    gender = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
    occupation = st.selectbox("Pekerjaan", ["Pelajar", "Mahasiswa", "Karyawan", "Pengangguran", "Lainnya"])

with col2:
    # Buat PHQ-9 lebih interaktif
    st.markdown("### 🩺 Skor PHQ-9 (Penilaian Depresi)")
    q1 = st.select_slider("Merasa sedih, murung, atau putus asa", ["Tidak pernah", "Jarang", "Sering"])
    q2 = st.select_slider("Kehilangan minat pada aktivitas", ["Tidak pernah", "Jarang", "Sering"])
    q3 = st.select_slider("Sulit beraktivitas sehari-hari", ["Tidak pernah", "Jarang", "Sering"])

    mapping = {"Tidak pernah": 0, "Jarang": 1, "Sering": 2}
    phq_score = mapping[q1] + mapping[q2] + mapping[q3]  # Simulasi skor PHQ sederhana

    sleep_hours = st.slider("Rata-rata Jam Tidur per Hari", 0, 12, 7)
    exercise_freq = st.slider("Frekuensi Olahraga per Minggu", 0, 7, 2)

st.divider()

# =========================================================
# Encoding Input
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

# === FIXED ===
# Samakan urutan dan jumlah kolom dengan model training
input_data = input_data.reindex(columns=model_columns, fill_value=0)

# =========================================================
# Prediksi
# =========================================================
st.subheader("🧾 Hasil Prediksi")
if st.button("🔍 Deteksi Tingkat Depresi"):
    try:
        prediction = model.predict(input_data)[0]
        result_label = le.inverse_transform([prediction])[0]
        
        if "depresi" in result_label.lower():
            st.error(f"🚨 Hasil Deteksi: **{result_label.upper()}**")
            st.markdown("> Disarankan untuk segera berkonsultasi dengan profesional kesehatan mental.")
        else:
            st.success(f"✅ Hasil Deteksi: **{result_label.upper()}**")
            st.markdown("> Kondisi baik, tetap jaga kesehatan mental dan fisik!")
    except Exception as e:
        st.error(f"Terjadi kesalahan dalam proses prediksi: {e}")

st.divider()
st.caption("© 2025 Mental Health Detection App — powered by Streamlit & scikit-learn")
