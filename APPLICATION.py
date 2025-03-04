import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import time
import warnings
import os

# Suppress warnings
warnings.filterwarnings('ignore')

# Install necessary libraries in the script itself
os.system("pip install streamlit numpy pandas matplotlib seaborn scikit-learn")

st.write("## Personal Fitness Tracker")
st.write("In this WebApp, you can predict the calories burned based on parameters such as `Age`, `Gender`, `BMI`, etc.")

st.sidebar.header("User Input Parameters")

def user_input_features():
    age = st.sidebar.slider("Age", 10, 100, 30)
    bmi = st.sidebar.slider("BMI", 15, 40, 20)
    duration = st.sidebar.slider("Duration (min)", 0, 35, 15)
    heart_rate = st.sidebar.slider("Heart Rate", 60, 130, 80)
    body_temp = st.sidebar.slider("Body Temperature (C)", 36, 42, 38)
    gender = 1 if st.sidebar.radio("Gender", ("Male", "Female")) == "Male" else 0
    
    return pd.DataFrame({
        "Age": [age], "BMI": [bmi], "Duration": [duration], 
        "Heart_Rate": [heart_rate], "Body_Temp": [body_temp], "Gender_male": [gender]
    })

df = user_input_features()

st.write("---")
st.header("Your Parameters")
progress_bar = st.progress(0)
for i in range(100):
    progress_bar.progress(i + 1)
    time.sleep(0.01)
st.write(df)

# Load data
calories = pd.read_csv("calories.csv")
exercise = pd.read_csv("exercise.csv")
exercise_df = exercise.merge(calories, on="User_ID").drop(columns="User_ID")

# Train-test split
train_data, test_data = train_test_split(exercise_df, test_size=0.2, random_state=1)
for data in [train_data, test_data]:
    data["BMI"] = round(data["Weight"] / ((data["Height"] / 100) ** 2), 2)

target = "Calories"
features = ["Gender", "Age", "BMI", "Duration", "Heart_Rate", "Body_Temp"]
train_data, test_data = train_data[features + [target]], test_data[features + [target]]
train_data, test_data = pd.get_dummies(train_data, drop_first=True), pd.get_dummies(test_data, drop_first=True)

# Prepare features
X_train, y_train = train_data.drop(columns=[target]), train_data[target]
X_test, y_test = test_data.drop(columns=[target]), test_data[target]

# Train model
model = RandomForestRegressor(n_estimators=1000, max_features=3, max_depth=6)
model.fit(X_train, y_train)

df = df.reindex(columns=X_train.columns, fill_value=0)
prediction = model.predict(df)

st.write("---")
st.header("Prediction")
progress_bar = st.progress(0)
for i in range(100):
    progress_bar.progress(i + 1)
    time.sleep(0.01)
st.write(f"{round(prediction[0], 2)} **kilocalories**")

st.write("---")
st.header("Similar Results")
similar_data = exercise_df[(exercise_df["Calories"].between(prediction[0] - 10, prediction[0] + 10))]
st.write(similar_data.sample(5))

st.write("---")
st.header("General Information")
st.write(f"You are older than {round((exercise_df['Age'] < df['Age'].values[0]).mean() * 100, 2)}% of other people.")
st.write(f"Your exercise duration is higher than {round((exercise_df['Duration'] < df['Duration'].values[0]).mean() * 100, 2)}% of other people.")
st.write(f"You have a higher heart rate than {round((exercise_df['Heart_Rate'] < df['Heart_Rate'].values[0]).mean() * 100, 2)}% of other people during exercise.")
st.write(f"You have a higher body temperature than {round((exercise_df['Body_Temp'] < df['Body_Temp'].values[0]).mean() * 100, 2)}% of other people during exercise.")
