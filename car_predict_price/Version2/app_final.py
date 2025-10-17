import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(layout="centered")

# -----------------------------
# Load model
# -----------------------------
model = joblib.load("/Users/klevizane/Documents/CienciaDatos/03_proyectos/02_car_predict price/Model/02_modelxgb.pkl")

# -----------------------------
# Sidebar: User input
# -----------------------------
def get_user_input():
    horsepower = st.number_input('Horsepower (No)', min_value=0, max_value=1000, step=1, value=300)
    torque = st.number_input('Torque (No)', min_value=0, max_value=1500, step=1, value=400)
    
    make = st.selectbox('Make', ['Aston Martin', 'Audi', 'BMW', 'Bentley', 'Ford', 'Mercedes-Benz', 'Nissan'])
    body_size = st.selectbox('Body Size', ['Compact', 'Large', 'Midsize'])
    body_style = st.selectbox('Body Style', [
        'Cargo Minivan', 'Cargo Van', 'Convertible', 'Convertible SUV', 'Coupe', 'Hatchback', 
        'Passenger Minivan', 'Passenger Van', 'Pickup Truck', 'SUV', 'Sedan', 'Wagon'
    ])
    engine_aspiration = st.selectbox('Engine Aspiration', [
        'Electric Motor', 'Naturally Aspirated', 'Supercharged', 'Turbocharged', 'Twin-Turbo', 'Twincharged'
    ])
    drivetrain = st.selectbox('Drivetrain', ['4WD', 'AWD', 'FWD', 'RWD'])
    transmission = st.selectbox('Transmission', ['automatic', 'manual'])
    
    user_data = {
        "Make": make,
        "Body Size": body_size,
        "Body Style": body_style,
        "Engine Aspiration": engine_aspiration,
        "Drivetrain": drivetrain,
        "Transmission": transmission,
        "Horsepower_no": horsepower,
        "Torque_no": torque,
    }
    return user_data

# App layout

st.markdown("<h1 style='text-align: center;'>🚗 Vehicle Price Prediction</h1>", unsafe_allow_html=True)

# Obtener datos del usuario
user_data = get_user_input()

# Función para preparar el input en formato DataFrame
def prepare_input(data, feature_list):
    df = pd.DataFrame([data], columns=feature_list)
    return df

# Lista de features originales
features = [
    "Make", "Body Size", "Body Style", "Engine Aspiration",
    "Drivetrain", "Transmission", "Horsepower_no", "Torque_no"
]

# Botón de predicción centrado
st.markdown("<br>", unsafe_allow_html=True)
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button("Predict Vehicle Price"):
        input_df = prepare_input(user_data, features)
        prediction = model.predict(input_df)
        st.success(f"**Predicted Price:** ${prediction[0]:,.2f}")

