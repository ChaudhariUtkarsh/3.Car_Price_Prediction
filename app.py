import os
import base64
import joblib
import streamlit as st
import pandas as pd


# 1. PAGE CONFIGURATION
st.set_page_config(page_title="Car Price Prediction", page_icon=" ", layout="wide", initial_sidebar_state="collapsed")


# 2. PROJECT PATHS
MODEL_PATH = "models/best_car_price_model.pkl"
BACKGROUND_PATH = "images/car_bg.jpg"


# 3. LOAD MODEL
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}. " "Run the Jupyter Notebook and save the trained model first.")
    return joblib.load(MODEL_PATH)
try:
    model = load_model()
except Exception as error:
    st.error(f"Unable to load model: {error}")
    st.stop()


# 4. DETECT MODEL FEATURES
try:
    preprocessor = model.named_steps["preprocessor"]
    model_features = list(preprocessor.feature_names_in_)
    categorical_encoder = (preprocessor.named_transformers_["cat"].named_steps["encoder"])
    categorical_columns = [col for col in model_features if col in ["brand", "fuel", "transmission", "mileage_unit"]]
    category_mapping = dict(zip(categorical_columns, categorical_encoder.categories_))
except Exception as error:
    st.error(f"Unable to detect model features: {error}")
    st.stop()

def get_options(column, default_options):
    """Get available categories from the trained encoder."""
    values = category_mapping.get(column)
    if values is None:
        return default_options
    options = [str(value) for value in values if pd.notna(value) and str(value).lower() != "nan"]
    return options if options else default_options


# 5. BACKGROUND IMAGE
def set_background(image_path):
    if not os.path.exists(image_path):
        st.warning("Background image not found. " "Add car_bg.jpg inside the images folder.")
        return
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode()
    extension = os.path.splitext(image_path)[1].lower()
    mime_type = ("image/png" if extension == ".png" else "image/jpeg")

    background_css = f"""
    <style>
    .stApp {{
        background:
        linear-gradient(
            rgba(0, 0, 0, 0.78),
            rgba(0, 0, 0, 0.82)
        ),
        url("data:{mime_type};base64,{encoded_image}");

        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}

    h1, h2, h3, p, label,
    [data-testid="stMarkdownContainer"] {{
        color: white !important;
    }}

    .main-title {{
        text-align: center;
        font-size: 42px;
        font-weight: 800;
        color: white;
        padding: 15px;
    }}

    .sub-title {{
        text-align: center;
        font-size: 17px;
        color: #dddddd;
        margin-bottom: 25px;
    }}

    .prediction-card {{
        background: rgba(255,255,255,0.95);
        padding: 25px;
        border-radius: 15px;
        text-align: center;
        margin-top: 20px;
    }}

    .prediction-card h2 {{
        color: #111111 !important;
        font-size: 24px;
    }}

    .prediction-price {{
        color: #087a36 !important;
        font-size: 35px;
        font-weight: bold;
    }}

    div.stButton > button {{
        width: 100%;
        background-color: #00a86b;
        color: white;
        font-size: 18px;
        font-weight: bold;
        border-radius: 10px;
        padding: 12px;
        border: none;
    }}

    div.stButton > button:hover {{
        background-color: #008f5a;
        color: white;
    }}

    </style>
    """
    st.markdown(background_css, unsafe_allow_html=True)
set_background(BACKGROUND_PATH)


# 6. APPLICATION HEADER
st.markdown('<div class="main-title"> Car Price Prediction</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">'
    'Predict the estimated selling price of your used car '
    'using Machine Learning'
    '</div>',
    unsafe_allow_html=True
)


# 7. CAR INPUT FORM
st.markdown("## Enter Your Car Details")
with st.form("car_prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        brand_options = get_options("brand", ["Maruti", "Hyundai", "Honda", "Tata"])
        brand = st.selectbox("Car Brand", options=brand_options)
        car_age = st.number_input("Car Age (Years)", min_value=0, max_value=100, value=5, step=1)
        km_driven = st.number_input("Kilometers Driven", min_value=0, max_value=1000000, value=45000, step=1000)

    with col2:
        fuel_options = get_options("fuel", ["petrol", "diesel", "cng", "lpg"])
        fuel = st.selectbox("Fuel Type", options=fuel_options)
        transmission_options = get_options("transmission", ["manual", "automatic"])
        transmission = st.selectbox("Transmission", options=transmission_options)
    mileage_value = None
    mileage_unit = None

    if "mileage_value" in model_features:
        st.markdown("### Mileage Details")
        mileage_value = st.number_input("Mileage (km/l or km/kg)", min_value=0.1, max_value=100.0, value=20.0, step=0.1)

    if "mileage_unit" in model_features:
        mileage_unit_options = get_options("mileage_unit", ["kmpl", "km/kg"])
        mileage_unit = st.selectbox("Mileage Unit", options=mileage_unit_options)
    predict_button = st.form_submit_button("Predict Car Price")


# 8. PRICE PREDICTION
if predict_button:
    input_data = {"brand": brand, "car_age": car_age, "km_driven": km_driven, "fuel": fuel, "transmission": transmission}
    if "mileage_value" in model_features:
        input_data["mileage_value"] = mileage_value
    if "mileage_unit" in model_features:
        input_data["mileage_unit"] = mileage_unit
    input_df = pd.DataFrame([input_data], columns=model_features)

    try:
        prediction = model.predict(input_df)[0]
        st.markdown(
            f"""
            <div class="prediction-card">
                <h2>Estimated Car Price</h2>
                <div class="prediction-price">
                    ₹ {prediction:,.2f}
                </div>
                <p style="color:#333333 !important;">
                    Your car price prediction is ready!
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("### Car Details")
        display_data = {"Car Brand": brand, "Car Age": f"{car_age} Years", "Kilometers Driven": f"{km_driven:,} km", "Fuel Type": fuel, "Transmission": transmission}
        if "mileage_value" in model_features:
            display_data["Mileage"] = (f"{mileage_value} {mileage_unit or ''}")
        st.table(pd.DataFrame(list(display_data.items()), columns=["Feature", "Value"]))
        st.info("Note: This is an estimated price generated by a Machine Learning model. Actual market prices may vary.")
    except Exception as error:
        st.error(f"Prediction failed: {error}")


# 9. ABOUT PROJECT
st.markdown("---")
st.markdown("## About This Project")
st.write(
    """
    This project predicts the selling price of used cars using Machine Learning regression algorithms.

    The model uses features such as:
    - Car Brand
    - Car Age
    - Kilometers Driven
    - Mileage (when available)
    - Fuel Type
    - Transmission

    The trained model is saved using Joblib and deployed through a Streamlit web application.
    """
)
st.caption("Developed as part of the Oasis Infobyte Data Science Internship.")