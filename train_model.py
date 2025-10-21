import pandas as pd
import pickle
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

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

X = df[['phq9_score','age','sex_Male','comorbidity_count','on_antidepressant']]
y = df['label_encoded']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = GaussianNB()
model.fit(X_train, y_train)

# Simpan semua komponen
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)
with open("model_columns.pkl", "wb") as f:
    pickle.dump(X_train.columns.tolist(), f)
