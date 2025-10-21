# ===============================
# app.py
# Mental Health Depression Detector
# ===============================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# =========================================================
# 🔹 LOAD MODEL, ENCODER, DAN URUTAN KOLOM
# =========================================================
@st.cache_resource
def load_model():
    try:
        base_path = os.path.dirname(__file__)
        model = pickle.load(open(os.path.join(base_path, "model.pkl"), "rb"))
        le = pickle.load(open(os.path.join(base_path, "label_encoder.pkl"), "rb"))
        model_columns = pickle.load(open(os.path.join(base_path, "model_columns.pkl"), "rb"))
        return model, le, model_columns
    except Exception as e:
        st.error("❌ Gagal memuat model atau encoder. Pastikan file 'model.pkl', 'label_encoder.pkl', dan 'model_columns.pkl' ada di folder yang sama.")
        st.stop()

model, le, model_columns = load_model()

# =========================================================
# 🧠 KONFIGURASI HALAMAN
# =========================================================
st.set_page_config(page_title="Mental Health Depression Detector", layout="centered")
st.title("🧠 Mental Health Depression Detector")

st.markdown("""
Aplikasi ini menggunakan model **Naive Bayes (GaussianNB)** untuk mendeteksi kemungkinan tingkat **depresi**  
berdasarkan data kesehatan mental dan hasil kuisioner **PHQ-9**.
""")

st.divider()

# =========================================================
# INPUT DATA USER
# =========================================================
st.subheader("Masukkan Data Responden")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Usia (tahun)", min_value=10, max_value=100, value=25)
    gender = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
    comorbidity_count = st.slider("Jumlah Penyakit Penyerta (komorbid)", 0, 5, 0)

with col2:
    on_antidepressant = st.selectbox("Apakah Mengonsumsi Antidepresan?", ["Tidak", "Ya"])
    st.markdown("### 🧩 Kuisioner PHQ-9 (Tiga Pertanyaan Ringkas)")
    q1 = st.selectbox("1️⃣ Apakah Anda sering merasa sedih, murung, atau putus asa?", ["Tidak pernah", "Jarang", "Sering"])
    q2 = st.selectbox("2️⃣ Apakah Anda kehilangan minat atau kesenangan dalam aktivitas sehari-hari?", ["Tidak pernah", "Jarang", "Sering"])
    q3 = st.selectbox("3️⃣ Apakah Anda mengalami kesulitan tidur atau tidur terlalu lama?", ["Tidak pernah", "Jarang", "Sering"])

# =========================================================
#  HITUNG SKOR PHQ-9 SEDERHANA
# =========================================================
phq_map = {"Tidak pernah": 0, "Jarang": 1, "Sering": 2}
phq_score = phq_map[q1] + phq_map[q2] + phq_map[q3]

st.markdown(f"** Skor PHQ-9 Anda:** `{phq_score}` (0–6)")

st.divider()

# =========================================================
# SIAPKAN DATA UNTUK PREDIKSI
# =========================================================
sex_Male = 1 if gender == "Laki-laki" else 0
on_antidepressant_val = 1 if on_antidepressant == "Ya" else 0

# Buat DataFrame input sesuai fitur training
input_data = pd.DataFrame([{
    'phq9_score': phq_score,
    'age': age,
    'sex_Male': sex_Male,
    'comorbidity_count': comorbidity_count,
    'on_antidepressant': on_antidepressant_val
}])

# Pastikan urutan kolom sama seperti saat training
try:
    input_data = input_data.reindex(columns=model_columns, fill_value=0)
except Exception as e:
    st.error(f"Gagal menyesuaikan urutan kolom: {e}")
    st.stop()

# Debugging (bisa dihapus nanti)
# st.write("Urutan kolom model:", model_columns)
# st.write("Urutan kolom input:", list(input_data.columns))

# =========================================================
# PROSES PREDIKSI
# =========================================================
st.subheader("Hasil Deteksi Depresi")

if st.button("Deteksi Tingkat Depresi"):
    try:
        prediction = model.predict(input_data)[0]
        result_label = le.inverse_transform([prediction])[0]

        st.write("---")

        if "berat" in result_label.lower():
            st.error(f"Hasil Deteksi: **{result_label.upper()}**")
            st.markdown("> Disarankan segera berkonsultasi dengan profesional kesehatan mental.")
        elif "sedang" in result_label.lower():
            st.warning(f"Hasil Deteksi: **{result_label.upper()}**")
            st.markdown("> Kamu menunjukkan gejala sedang, coba jaga pola tidur, olahraga, dan aktivitas sosial.")
        else:
            st.success(f"Hasil Deteksi: **{result_label.upper()}**")
            st.markdown("> Kesehatan mentalmu tampak baik! Pertahankan gaya hidup seimbang.")
    except Exception as e:
        st.error(f"Terjadi kesalahan dalam proses prediksi: {e}")

st.divider()

# =========================================================
# INFO TAMBAHAN
# =========================================================
st.subheader("Tentang Aplikasi")
st.markdown("""
- Model: **Naive Bayes (GaussianNB)**
- Fitur utama: PHQ-9, usia, jenis kelamin, komorbiditas, konsumsi antidepresan  
- Dataset: Nasional (versi praproses)
- Tujuan: Edukasi & penelitian non-medis
""")

st.caption("© 2025 Mental Health Detection App — powered by Streamlit & scikit-learn")
