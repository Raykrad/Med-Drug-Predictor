import streamlit as st
import pandas as pd
import joblib

model = joblib.load('saved_model.pkl')
le_dict = joblib.load('label_encoders.pkl')

st.set_page_config(page_title="Drug Predictor", layout="centered")
st.title("Предсказание лекарственного препарата")

age = st.slider("Возраст", 15, 75, 35)
sex = st.selectbox("Пол", ['F', 'M'])
bp = st.selectbox("Давление", ['LOW', 'NORMAL', 'HIGH'])
chol = st.selectbox("Холестерин", ['NORMAL', 'HIGH'])
na_to_k = st.slider("Na_to_K (соотношение натрий/калий)", 5.0, 40.0, 10.0)

input_data = pd.DataFrame([[age, sex, bp, chol, na_to_k]],
                          columns=['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K'])

for col in ['Sex', 'BP', 'Cholesterol']:
    le = le_dict[col]
    input_data[col] = le.transform(input_data[col])

if st.button("Предсказать препарат"):
    prediction = model.predict(input_data)[0]
    st.success(f"Рекомендованный препарат: **{prediction}**")
