# ===============================
# app.py
# ===============================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# =========================================================
# LOAD MODEL DAN ENCODER
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
# KONFIGURASI HALAMAN
# =========================================================
st.set_page_config(page_title="Mental Health Detector", layout="centered")
st.title("🧠 Mental Health Depression Detector")
st.markdown("""
Aplikasi ini menggunakan model *Naive Bayes* untuk memprediksi tingkat kemungkinan **depresi**  
berdasarkan data kesehatan mental dan hasil **PHQ-9**.
""")

st.divider()

# =========================================================
# INPUT DATA USER
# =========================================================
st.subheader("📋 Masukkan Data Responden")

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Usia", min_value=10, max_value=100, value=25)
    gender = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
    comorbidity_count = st.slider("Jumlah Penyakit Penyerta", 0, 5, 0)
with col2:
    on_antidepressant = st.selectbox("Apakah Mengonsumsi Antidepresan?", ["Tidak", "Ya"])
    st.markdown("### 🧩 Kuesioner PHQ-9")
    q1 = st.selectbox("1️⃣ Merasa sedih, murung, atau putus asa:", ["Tidak pernah", "Jarang", "Sering"])
    q2 = st.selectbox("2️⃣ Kehilangan minat atau kesenangan dalam aktivitas:", ["Tidak pernah", "Jarang", "Sering"])
    q3 = st.selectbox("3️⃣ Kesulitan tidur atau tidur terlalu lama:", ["Tidak pernah", "Jarang", "Sering"])

# =========================================================
# HITUNG SKOR PHQ-9 SEDERHANA
# =========================================================
phq_map = {"Tidak pernah": 0, "Jarang": 1, "Sering": 2}
phq_score = phq_map[q1] + phq_map[q2] + phq_map[q3]

# =========================================================
# ENCODING FITUR
# =========================================================
sex_Male = 1 if gender == "Laki-laki" else 0
on_antidepressant_val = 1 if on_antidepressant == "Ya" else 0

input_data = pd.DataFrame([{
    "age": age,
    "sex_Male": sex_Male,
    "comorbidity_count": comorbidity_count,
    "on_antidepressant": on_antidepressant_val,
    "phq9_score": phq_score
}])

# Pastikan urutan kolom sama seperti saat training
input_data = input_data.reindex(columns=model_columns, fill_value=0)

st.divider()
st.subheader("📊 Hasil Prediksi")

# =========================================================
# PREDIKSI
# =========================================================
if st.button("🔍 Deteksi Tingkat Depresi"):
    try:
        prediction = model.predict(input_data)[0]
        result_label = le.inverse_transform([prediction])[0]

        if "berat" in result_label.lower():
            st.error(f"🚨 Hasil Deteksi: **{result_label.upper()}**")
            st.markdown("> Disarankan segera konsultasi dengan profesional kesehatan mental.")
        elif "sedang" in result_label.lower():
            st.warning(f"⚠️ Hasil Deteksi: **{result_label.upper()}**")
            st.markdown("> Kamu menunjukkan gejala sedang, coba tingkatkan rutinitas positif dan istirahat cukup.")
        else:
            st.success(f"✅ Hasil Deteksi: **{result_label.upper()}**")
            st.markdown("> Kesehatan mentalmu tampak baik, pertahankan keseimbangan hidupmu!")
    except Exception as e:
        st.error(f"Terjadi kesalahan dalam proses prediksi: {e}")

st.divider()
st.caption("© 2025 Mental Health Detection App — powered by Streamlit & scikit-learn")
