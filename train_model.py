# ===============================
# train_model.py
# ===============================

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# -------------------------------------------------------
# 1️⃣ CONTOH DATA — ganti bagian ini dengan dataset kamu
# -------------------------------------------------------
# Contoh sederhana untuk testing
data = {
    'age': [22, 35, 45, 28, 52, 33, 41],
    'sex_Male': [1, 0, 1, 1, 0, 1, 0],
    'comorbidity_count': [0, 1, 2, 0, 3, 1, 2],
    'on_antidepressant': [0, 0, 1, 0, 1, 0, 1],
    'phq9_score': [5, 15, 23, 9, 25, 13, 19],
    'label': ['normal', 'ringan', 'berat', 'normal', 'berat', 'sedang', 'sedang']
}

df = pd.DataFrame(data)

# -------------------------------------------------------
# 2️⃣ ENCODING LABEL TARGET
# -------------------------------------------------------
le = LabelEncoder()
df['label_encoded'] = le.fit_transform(df['label'])

# -------------------------------------------------------
# 3️⃣ SPLIT DATA
# -------------------------------------------------------
X = df.drop(columns=['label', 'label_encoded'])
y = df['label_encoded']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------------------------------------
# 4️⃣ LATIH MODEL NAIVE BAYES
# -------------------------------------------------------
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

# -------------------------------------------------------
# 5️⃣ EVALUASI SINGKAT
# -------------------------------------------------------
y_pred = nb_model.predict(X_test)
print("Akurasi:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=le.classes_))

# -------------------------------------------------------
# 6️⃣ SIMPAN MODEL, ENCODER, DAN KOLOM
# -------------------------------------------------------
with open("model.pkl", "wb") as f:
    pickle.dump(nb_model, f)

with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

model_columns = X_train.columns.tolist()
with open("model_columns.pkl", "wb") as f:
    pickle.dump(model_columns, f)

print("\n✅ Model, label encoder, dan kolom fitur berhasil disimpan!")
