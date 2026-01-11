import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Country GDP Prediction", layout="centered")
st.title("🌍 Country Data Prediction App")

# Load data
@st.cache_data
def load_data():
    path = r"C:\Users\Aathira\Desktop\Assingment 2\country_data.csv"
    return pd.read_csv(path)

df = load_data()

st.subheader("📊 Dataset Preview")
st.dataframe(df.head())

# Drop non-numeric & target
features = df.drop(columns=["country", "gdpp"])
target = df["gdpp"]

# Train model
X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

st.subheader("🧠 Enter Values to Predict GDP per Capita")

# User inputs
child_mort = st.number_input("Child Mortality", min_value=0.0)
exports = st.number_input("Exports")
health = st.number_input("Health Spending")
imports = st.number_input("Imports")
income = st.number_input("Income")
inflation = st.number_input("Inflation")
life_expec = st.number_input("Life Expectancy")
total_fer = st.number_input("Total Fertility")
gdpp_dummy = 0  # placeholder

if st.button("🔮 Predict GDP"):
    input_data = [[
        child_mort, exports, health, imports,
        income, inflation, life_expec, total_fer
    ]]
    prediction = model.predict(input_data)
    st.success(f"💰 Predicted GDP per Capita: {int(prediction[0])}")
