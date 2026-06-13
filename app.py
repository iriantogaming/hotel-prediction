
import streamlit as st
import joblib
import numpy as np
from datetime import date

model = joblib.load("hotel_model.pkl")
room_encoder = joblib.load("room_encoder.pkl")

st.title("Hotel Room Price Prediction")

st.write(
    "Prediksi harga kamar hotel berdasarkan tanggal check-in, jumlah tamu, lama menginap, dan tipe kamar."
)

arrival_date = st.date_input(
    "Arrival Date",
    value=date.today()
)

weekend_nights = st.number_input(
    "Weekend Nights",
    min_value=0,
    value=1
)

week_nights = st.number_input(
    "Weekday Nights",
    min_value=0,
    value=1
)

total_guests = st.number_input(
    "Total Guests",
    min_value=1,
    value=2
)

room_type = st.selectbox(
    "Room Type",
    ["A", "B", "C", "D", "E", "F", "G"]
)

if st.button("Predict Price"):

    arrival_year = arrival_date.year
    arrival_month_num = arrival_date.month
    arrival_day = arrival_date.day

    total_nights = weekend_nights + week_nights

    day_of_week = arrival_date.weekday()

    is_weekend = 1 if day_of_week >= 5 else 0

    room_encoded = room_encoder.transform([room_type])[0]

    data = np.array([[
        arrival_year,
        arrival_month_num,
        arrival_day,
        weekend_nights,
        week_nights,
        total_guests,
        room_encoded,
        total_nights,
        is_weekend
    ]])

    prediction = model.predict(data)

    st.success(
        f"Predicted ADR = {prediction[0]:.2f}"
    )
