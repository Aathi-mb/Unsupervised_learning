import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Country Data Prediction App",
    layout="centered"
)

st.title("🌍 Country Data Prediction App")
st.write("Predict GDP per Capita using Machine Learning")

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    return pd.read_csv("country_data.csv")

df = load_data()

# ---------------- DATA PREVIEW ----------------
st.subheader("📊 Dataset Preview")
st.dataframe(df.head())

st.subheader("ℹ Dataset Info")
st.write("Rows:", df.shape[0])
st.write("Columns:", df.shape[1])

# ---------------- MODEL PREPARATION ----------------
# Drop non-numeric column
X = df.drop(columns=["country", "gdpp"])
y = df["gdpp"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
score = r2_score(y_test, y_pred)

st.subheader("📈 Model Performance")
st.write(f"R² Score: **{score:.2f}**")

# ---------------- USER INPUT ----------------
st.subheader("🧠 Enter Values to Predict GDP per Capita")

child_mort = st.number_input("Child Mortality", min_value=0.0)
exports = st.number_input("Exports")
health = st.number_input("Health Spending")
imports = st.number_input("Imports")
income = st.number_input("Income")
inflation = st.number_input("Inflation")
life_expec = st.number_input("Life Expectancy")
total_fer = st.number_input("Total Fertility")

# ---------------- PREDICTION ----------------
if st.button("🔮 Predict GDP"):
    input_data = [[
        child_mort,
        exports,
        health,
        imports,
        income,
        inflation,
        life_expec,
        total_fer
    ]]

    prediction = model.predict(input_data)
    st.success(f"💰 Predicted GDP per Capita: {int(prediction[0])}")

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("Built with ❤️ using **Streamlit & Machine Learning**")
