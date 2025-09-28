import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler # Estandarizar



model = joblib.load("Model/model.pkl")
scaler = joblib.load("Model/scaler.pkl")

st.title("Iris prediction")

# Valores a elegir
# (valor_mínimo, valor_máximo, valor_inicial)
sepal_length = st.slider("sepal length (cm)", 4.0, 8.0, 5.0)
sepal_width = st.slider("sepal width (cm)", 2.0, 4.5, 3.0)
petal_length = st.slider("petal length (cm)", 1.0, 7.0, 4.0)
petal_width = st.slider("petal width (cm)", 0.1, 2.5, 1.0)


# Preparar datos introducidos por el usuario
input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]]) # Estandarizar no se hacer eso de momento

# estandarizar los datos
input_data_scaled = scaler.transform(input_data) # Estandarizamos los datos

# Hacer prediccion
prediction = model.predict(input_data_scaled)


# Mostrar prediccion
predicted_class = {0: "setosa", 1: "versicolor", 2: "virginica"}[prediction[0]]
st.write(f"Predicted class: {predicted_class}")



