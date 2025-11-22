import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from utils.helpers import load_db_table
from utils.config import LSTM_MODEL_PATH

st.title("🌍 Real-Time Greenhouse Gas Monitoring & Prediction")

df = load_db_table()
df["timestamp"] = pd.to_datetime(df["timestamp"])

st.subheader("📡 Live GHG Data")
st.write(df.tail())

st.line_chart(df.set_index("timestamp")[["co", "no2", "o3"]])

# Load model
model = tf.keras.models.load_model(LSTM_MODEL_PATH)

series = df["co"].values.reshape(-1, 1)
scaler = MinMaxScaler()
series_scaled = scaler.fit_transform(series)

window = 24
X_input = series_scaled[-window:].reshape(1, window, 1)

pred_scaled = model.predict(X_input)
prediction = float(scaler.inverse_transform(pred_scaled))

st.subheader("🔮 Predicted CO for next hour:")
st.metric(label="CO (mg/m³)", value=prediction)
