import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.express as px
from PIL import Image

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(layout="wide")

# -----------------------------
# Load model and data
# -----------------------------
model = joblib.load("/Users/klevizane/Documents/CienciaDatos/03_proyectos/02_car_predict price/Model/02_modelxgb.pkl")

final_fi = pd.read_csv("/Users/klevizane/Documents/CienciaDatos/03_proyectos/02_car_predict price/data/feature_importance.csv", index_col=0)

# -----------------------------
# Sidebar: User input
# -----------------------------
def get_user_input():
    horsepower = st.sidebar.number_input('Horsepower (No)', min_value=0, max_value=1000, step=1, value=300)
    torque = st.sidebar.number_input('Torque (No)', min_value=0, max_value=1500, step=1, value=400)
    
    make = st.sidebar.selectbox('Make', ['Aston Martin', 'Audi', 'BMW', 'Bentley', 'Ford', 'Mercedes-Benz', 'Nissan'])
    body_size = st.sidebar.selectbox('Body Size', ['Compact', 'Large', 'Midsize'])
    body_style = st.sidebar.selectbox('Body Style', [
        'Cargo Minivan', 'Cargo Van', 'Convertible', 'Convertible SUV', 'Coupe', 'Hatchback', 
        'Passenger Minivan', 'Passenger Van', 'Pickup Truck', 'SUV', 'Sedan', 'Wagon'
    ])
    engine_aspiration = st.sidebar.selectbox('Engine Aspiration', [
        'Electric Motor', 'Naturally Aspirated', 'Supercharged', 'Turbocharged', 'Twin-Turbo', 'Twincharged'
    ])
    drivetrain = st.sidebar.selectbox('Drivetrain', ['4WD', 'AWD', 'FWD', 'RWD'])
    transmission = st.sidebar.selectbox('Transmission', ['automatic', 'manual'])
    
    # Aquí ya no hacemos one-hot encoding en el input del usuario directamente por el pipeline que tiene el modelo
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


st.markdown("<h1 style='text-align: center;'>🚗 Vehicle Price Prediction App</h1>", unsafe_allow_html=True)

left_col, right_col = st.columns(2)

# Left column: Feature importance

"""with left_col:
    st.header("Feature Importance")
    
    final_fi_sorted = final_fi.sort_values(by='Feature Importance Score', ascending=True)
    
    fig = px.bar(
        final_fi_sorted,
        x='Feature Importance Score',
        y='Variable',
        orientation='h',
        title="Feature Importance",
        labels={'Feature Importance Score': 'Importance', 'Variable': 'Feature'},
        text='Feature Importance Score',
        color_discrete_sequence=['#48a3b4']
    )
    fig.update_layout(
        xaxis_title="Feature Importance Score",
        yaxis_title="Variable",
        template="plotly_white",
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)
"""
# Right column: Prediction
with right_col:
    st.header("Predict Vehicle Price")

    # Obtener datos del usuario
    user_data = get_user_input()

    # Función para preparar el input en formato DataFrame
    def prepare_input(data, feature_list):
        df = pd.DataFrame([data], columns=feature_list)
        return df

    # Lista de features originales (sin one-hot)
    features = [
        "Make", "Body Size", "Body Style", "Engine Aspiration",
        "Drivetrain", "Transmission", "Horsepower_no", "Torque_no"
    ]

    # Botón para predecir
    if st.button("Predict"):
        input_df = prepare_input(user_data, features)
        prediction = model.predict(input_df)

        st.subheader("Predicted Price")
        st.write(f"${prediction[0]:,.2f}")

# -----------------------------
# Run en terminal:
# streamlit run "/Users/klevizane/Documents/CienciaDatos/03_proyectos/02_car_predict price/Version2/app2.2.py"
# -----------------------------
