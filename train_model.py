# train_model.py (potongan penting)
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder

# --- load atau buat DataFrame df ---
# df harus berisi semua fitur yg nanti dipakai; contoh di bawah hanya ilustrasi
# df = pd.read_csv('data/your_data.csv')

# contoh sederhana:
data = {
    'age':[25,40,30],
    'sex_Male':[1,0,1],
    'comorbidity_count':[0,1,0],
    'on_antidepressant':[0,1,0],
    'phq9_score':[5,14,20],
    'label':['normal','moderate','severe']
}
df = pd.DataFrame(data)

le = LabelEncoder()
df['label_encoded'] = le.fit_transform(df['label'])

X = df[['phq9_score','age','sex_Male','comorbidity_count','on_antidepressant']]  # PASTIKAN urutan kolom di sini
y = df['label_encoded']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)

model = GaussianNB()
model.fit(X_train, y_train)

# simpan model, encoder, dan urutan kolom
with open('model.pkl','wb') as f:
    pickle.dump(model, f)
with open('label_encoder.pkl','wb') as f:
    pickle.dump(le, f)
model_columns = X_train.columns.tolist()
with open('model_columns.pkl','wb') as f:
    pickle.dump(model_columns, f)

print('Saved model.pkl, label_encoder.pkl, model_columns.pkl')
print('Model expected feature order:', model_columns)
# optional: print feature names stored in model (if available)
try:
    print('model.feature_names_in_ :', model.feature_names_in_)
except Exception:
    pass
