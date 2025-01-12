from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Charger le modèle préalablement sauvegardé
model = joblib.load('lgb_model.pkl')

# Seuil optimal pour la classification
seuil_optimal = 0.49  

# Initialiser FastAPI
app = FastAPI()

# Définir un message d'accueil personnalisé
@app.get("/")
def read_root():
    return {"message": "Bienvenue à l'API de prédiction de scoring de crédit"}


# Définir la structure des données d'entrée
class InputData(BaseModel):
    EXT_SOURCE_2: float
    DAYS_BIRTH: int
    EXT_SOURCE_3: float
    bureau_DAYS_CREDIT_max: float
    bureau_DAYS_CREDIT_min: float
    bureau_DAYS_CREDIT_UPDATE_mean: float
    bureau_DAYS_CREDIT_mean: float
    bureau_CREDIT_ACTIVE_Closed_mean: float
    SK_ID_CURR: int
    bureau_CREDIT_ACTIVE_Active_mean: float

# Définir l'endpoint de prédiction
# @app.post("/predict/")
# def predict(input_data: InputData):
#     # Convertir les données d'entrée en numpy array
#     data = np.array([[
#         input_data.EXT_SOURCE_2, input_data.DAYS_BIRTH, input_data.EXT_SOURCE_3, 
#         input_data.bureau_DAYS_CREDIT_max, input_data.bureau_DAYS_CREDIT_min, 
#         input_data.bureau_DAYS_CREDIT_UPDATE_mean, input_data.bureau_DAYS_CREDIT_mean, 
#         input_data.bureau_CREDIT_ACTIVE_Closed_mean, input_data.SK_ID_CURR, 
#         input_data.bureau_CREDIT_ACTIVE_Active_mean
#     ]])

#     # Faire la prédiction
#     prediction = model.predict(data)
#     proba = model.predict_proba(data)[:, 1]  # Probabilité de la classe 1

#     # Retourner la prédiction et la probabilité
#     return {"prediction": int(prediction[0]), "probability": float(proba[0])}

# Fonction pour catégoriser les clients en trois segments basés sur leur probabilité
def classify_client(probability):
    if probability >= 0.75:
        return "Client à haut risque : Le crédit a très peu de chances d'être approuvé."
    elif 0.5 <= probability < 0.75:
        return "Client à risque modéré : Une étude plus approfondie peut être nécessaire avant une décision finale."
    else:
        return "Client fiable : Le crédit est susceptible d'être approuvé."

# 
@app.post("/predict")
def predict(input_data: InputData):
    # Convertir les données d'entrée en tableau numpy
    data = np.array([[
        input_data.EXT_SOURCE_2,
        input_data.DAYS_BIRTH,
        input_data.EXT_SOURCE_3,
        input_data.bureau_DAYS_CREDIT_max,
        input_data.bureau_DAYS_CREDIT_min,
        input_data.bureau_DAYS_CREDIT_UPDATE_mean,
        input_data.bureau_DAYS_CREDIT_mean,
        input_data.bureau_CREDIT_ACTIVE_Closed_mean,
        input_data.SK_ID_CURR,
        input_data.bureau_CREDIT_ACTIVE_Active_mean
    ]])

    # Prédiction des probabilités
    probabilities = model.predict_proba(data)[:, 1]
    prediction = (probabilities >= seuil_optimal).astype(int)

    # Générer le message de risque basé sur la probabilité
    risk_message = classify_client(probabilities[0])

    # Retourner le résultat au client
    return {
        "probability": round(probabilities[0], 2),
        "prediction": int(prediction),
        "message": "1 = Client à risque (crédit non approuvé), 0 = Client fiable (crédit approuvé)",
        "message de risque": risk_message
    }

# Lancer l'API avec uvicorn API:app --reload
# Exemple d'utilisation avec json:
# {
#   "EXT_SOURCE_2": 0.5,
#   "DAYS_BIRTH": -12000,
#   "EXT_SOURCE_3": 0.7,
#   "bureau_DAYS_CREDIT_max": -1000,
#   "bureau_DAYS_CREDIT_min": -2000,
#   "bureau_DAYS_CREDIT_UPDATE_mean": -1500,
#   "bureau_DAYS_CREDIT_mean": -1750,
#   "bureau_CREDIT_ACTIVE_Closed_mean": 0.2,
#   "SK_ID_CURR": 312732,
#   "bureau_CREDIT_ACTIVE_Active_mean": 0.3
# }



