# app.py (replace contents with this)
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from helpers import fetch_location_air_quality, extract_latest_pollutant
from config import LSTM_MODEL_PATH

st.set_page_config(page_title="GHG Monitor (Location-based)", layout="wide")

st.title("🌍 Real-Time GHG Monitoring — Location Based")

# --- 1) Try to obtain browser geolocation (optional package) ---
lat = None
lon = None
geoloc_installed = False

try:
    # streamlit-geolocation is optional; if installed we use it for automatic browser geolocation
    from streamlit_geolocation import geolocation
    geoloc_installed = True
except Exception:
    geoloc_installed = False

col1, col2 = st.columns([1, 2])

with col1:
    st.markdown("### Location input")
    if geoloc_installed:
        st.markdown("Click the button below to allow the browser to share your precise location.")
        if st.button("Use my browser location"):
            try:
                location = geolocation()  # will prompt browser permission
                if location and "latitude" in location and "longitude" in location:
                    lat = location["latitude"]
                    lon = location["longitude"]
                else:
                    st.warning("Location permission denied or unavailable. Please enter coordinates manually.")
            except Exception as e:
                st.warning("Automatic geolocation failed: " + str(e))
    else:
        st.info("Automatic browser location not available. (Install `streamlit-geolocation` to enable auto-detect.)")

    # Manual fallback
    st.markdown("**Or enter coordinates manually (latitude, longitude)**")
    lat_in = st.text_input("Latitude (e.g. 22.5726)", value="")
    lon_in = st.text_input("Longitude (e.g. 88.3639)", value="")

    # If browser geolocation gave values, show them and allow override
    if lat is not None and lon is not None:
        st.success(f"Detected: {lat:.6f}, {lon:.6f}")
        # allow user to override if they want:
        if st.checkbox("Override detected coordinates?"):
            lat = float(st.text_input("Latitude (override)", value=str(lat)))
            lon = float(st.text_input("Longitude (override)", value=str(lon)))
    else:
        # if manual inputs provided, use them
        if lat_in.strip() != "" and lon_in.strip() != "":
            try:
                lat = float(lat_in)
                lon = float(lon_in)
            except ValueError:
                st.error("Invalid manual coordinates. Please enter numeric latitude and longitude.")

    # Button to fetch data for the chosen coordinates
    fetch_button = st.button("Fetch data for this location")

with col2:
    st.markdown("### Map preview")
    if lat is not None and lon is not None:
        map_df = pd.DataFrame({"lat":[lat], "lon":[lon]})
        st.map(map_df)
    else:
        st.info("Map will show selected location after coordinates are provided.")

# If user clicked fetch (or we already have coords), get data
if fetch_button or (lat is not None and lon is not None):
    if lat is None or lon is None:
        st.error("Coordinates missing. Please allow location access or enter latitude and longitude.")
    else:
        st.success(f"Fetching hyperlocal air quality for {lat:.6f}, {lon:.6f} ...")
        try:
            # hourly_vars can be adjusted. We ensure ozone (o3) is fetched for the pollutant section.
            hourly_vars = [
                "pm10","pm2_5","carbon_monoxide","nitrogen_dioxide",
                "sulphur_dioxide","ozone","temperature_2m",
                "relative_humidity_2m","wind_speed_10m"
            ]
            df_local = fetch_location_air_quality(lat, lon, hourly_vars=hourly_vars)

            # Show raw tail
            st.subheader("📡 Live GHG & Air Quality Data (area)")
            st.write(df_local.tail(6))

            # Overall charts (CO, NO2, O3)
            st.subheader("Trends — CO, NO₂, O₃ (local)")
            chart_cols = ["co", "no2", "o3"]
            present_cols = [c for c in chart_cols if c in df_local.columns]
            st.line_chart(df_local.set_index("timestamp")[present_cols])

            # Two-column layout: left = main pollutants, right = O3 highlighted
            left, right = st.columns([2, 1])

            with left:
                st.subheader("Other pollutants")
                other_cols = [c for c in ["pm10","pm2_5","so2","temp","humidity","wind_speed"] if c in df_local.columns]
                if other_cols:
                    st.line_chart(df_local.set_index("timestamp")[other_cols])
                else:
                    st.write("No additional pollutant columns available.")

            # ------------------ O3 Section (Environment pollutant) ------------------
            with right:
                st.markdown("## 🧪 Environment pollutant — O₃")
                o3_value, o3_time = extract_latest_pollutant(df_local, "o3")
                if o3_value is None:
                    st.write("O₃ data not available for this location/time.")
                else:
                    st.metric(label="Latest O₃ (µg/m³ or API units)", value=f"{o3_value:.3f}", delta=None)
                    # small O3 chart
                    if "o3" in df_local.columns:
                        st.line_chart(df_local.set_index("timestamp")[["o3"]])
                    # Simple O3 trend comment
                    if len(df_local.dropna(subset=["o3"])) >= 3:
                        last3 = df_local.dropna(subset=["o3"]).sort_values("timestamp")["o3"].iloc[-3:].values
                        trend = "up" if last3[-1] > last3[0] else "down" if last3[-1] < last3[0] else "stable"
                        st.write(f"Recent short-term trend: **{trend}** (last 3 hourly samples)")

            st.markdown("---")

            # ------------------ LSTM Prediction for CO (reuse your logic) ------------------
            if "co" in df_local.columns:
                st.subheader("🔮 Location-based CO prediction (next hour)")
                series = df_local["co"].dropna().values.reshape(-1, 1)
                if len(series) < 30:
                    st.warning("Not enough local historical points to produce a reliable prediction. Collect more data (run fetch hourly).")
                else:
                    # scale and prepare input window (same logic as you used)
                    scaler = MinMaxScaler()
                    series_scaled = scaler.fit_transform(series)
                    window = 24
                    if len(series_scaled) <= window:
                        st.warning("Need more than window size points to predict.")
                    else:
                        X_input = series_scaled[-window:].reshape(1, window, 1)
                        # load model safely
                        try:
                            model = tf.keras.models.load_model(LSTM_MODEL_PATH, compile=False)
                            pred_scaled = model.predict(X_input)
                            pred = float(scaler.inverse_transform(pred_scaled)[0][0])
                            st.metric("Predicted CO (next hour)", f"{pred:.3f}")
                        except Exception as e:
                            st.error("Could not load LSTM model or predict: " + str(e))

        except Exception as e:
            st.error("Failed to fetch or process local data: " + str(e))

else:
    st.info("Please click 'Use my browser location' or enter coordinates and press 'Fetch data for this location' to view local results.")
