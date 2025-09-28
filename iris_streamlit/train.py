

import pandas as pd
from sklearn.preprocessing import StandardScaler # Estandarizar
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
import numpy as np
import joblib



df = pd.read_csv("/Users/klevizane/Documents/CienciaDatos/03_proyectos/iris_streamlit/Data/iris.csv")


# Separamos os datos
X = df.drop(columns=["target"])
y = df["target"]

# Creamos X_train, X_test, y_train, y_test 
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=11, test_size=0.2)

# Escalar datos
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# entrenar modelo
model = SVC(C=1)
model.fit(X_train, y_train)


# Vemos el score
joblib.dump(model, "Model/model.pkl")
joblib.dump(scaler, "Model/scaler.pkl")




"""# Creamos el pipeline
scaler = StandardScaler()
svc = SVC(C=1)
pipe = make_pipeline(scaler, svc)"""