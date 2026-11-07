import streamlit as st
import pandas as pd
import joblib
import os
import base64

# Page Configuration
st.set_page_config(page_title="Car Price Prediction", page_icon=" ", layout="centered")

# Background Image Function
def set_background(image_path):
    with open(image_path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image:
                linear-gradient(rgba(0,0,0,0.65), rgba(0,0,0,0.65)),
                url("data:image/jpeg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            background-repeat: no-repeat;
        }}

        h1, h2, h3, p, label {{
            color: white !important;
        }}

        [data-testid="stMetricValue"] {{
            color: #00ff99 !important;
        }}

        .stButton button {{
            background-color: #ff6600;
            color: white;
            border-radius: 10px;
            border: none;
            font-weight: bold;
        }}

        .stButton button:hover {{
            background-color: #e65c00;
            color: white;
        }}

        </style>
        """,
        unsafe_allow_html=True
    )

# Apply Background Image
BACKGROUND_PATH = "images/car_bg.jpg"
if os.path.exists(BACKGROUND_PATH):
    set_background(BACKGROUND_PATH)
else:
    st.warning("Background image not found. Check images/car_bg.jpg")


# Load Machine Learning Model
MODEL_PATH = "models/best_car_price_model.pkl"
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)
if not os.path.exists(MODEL_PATH):
    st.error("Model file not found! Check models folder.")
    st.stop()
model = load_model()


# Application Title
st.title("Car Price Prediction System")
st.write("Predict the estimated selling price of a used car.")
st.divider()


# User Inputs
st.subheader("Enter Car Details")
brand = st.selectbox(
    "Select Car Brand",
    ["Maruti", "Hyundai", "Honda", "Toyota", "Tata", "Mahindra", "Ford", "BMW", "Renault", "Volkswagen", "Kia", "Other"]
)

car_age = st.number_input("Car Age (Years)", min_value=0, max_value=40, value=5)
km_driven = st.number_input("Kilometers Driven", min_value=0, max_value=1000000, value=45000)
fuel = st.selectbox("Fuel Type", ["petrol", "diesel", "cng", "lpg", "electric"])
transmission = st.selectbox("Transmission", ["manual", "automatic"])


# Prediction
if st.button("Predict Car Price", use_container_width=True):
    input_data = pd.DataFrame([{"brand": brand, "car_age": car_age, "km_driven": km_driven, "fuel": fuel, "transmission": transmission}])
    prediction = model.predict(input_data)[0]
    st.success("Prediction Completed!")
    st.metric("Estimated Car Price", f"{prediction:,.0f}")
    st.info("This is an estimated price based on the trained Machine Learning model.")

# Footer
st.divider()
st.caption("Car Price Prediction | Machine Learning Project")