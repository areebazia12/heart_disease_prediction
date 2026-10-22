
import pickle
import pandas as pd
import streamlit as st

FEATURES = ["age", "sex", "cp", "thalach"]

@st.cache_resource
def load_model():
    with open("heart_disease_model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

st.title("Heart Disease Prediction System")
st.write(
    "Educational machine-learning demo using "
    "four UCI Heart Disease attributes."
)

age = st.number_input(
    "Age (years)",
    min_value=1,
    max_value=120,
    value=52
)

sex_label = st.selectbox(
    "Sex",
    ["Male (1)", "Female (0)"]
)
sex = 1 if sex_label.startswith("Male") else 0

cp = st.selectbox(
    "Chest pain type (UCI code)",
    [1, 2, 3, 4],
    index=3
)

thalach = st.number_input(
    "Maximum heart rate achieved",
    min_value=40,
    max_value=250,
    value=150
)

if st.button("Predict"):
    row = pd.DataFrame(
        [[age, sex, cp, thalach]],
        columns=FEATURES
    )

    result = int(model.predict(row)[0])

    if result == 1:
        st.warning("Model output: Heart Disease class (1)")
    else:
        st.success("Model output: No Heart Disease class (0)")

    st.caption(
        "Educational demonstration only. "
        "This is not a medical diagnosis."
    )
