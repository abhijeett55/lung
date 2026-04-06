import os
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

MODEL_PATH = os.path.join(os.path.dirname(__file__), "lung_model.pkl")
model = joblib.load(MODEL_PATH)

print("✅ Model loaded:", type(model))


# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Input schema
class InputData(BaseModel):
    smoking: int
    yellow_fingers: int
    anxiety: int
    peer_pressure: int
    chronic_disease: int
    fatigue: int
    allergy: int
    wheezing: int
    alcohol_consuming: int
    coughing: int
    shortness_of_breath: int
    swallowing_difficulty: int
    chest_pain: int

@app.get("/")
def home():
    return {"message": "Lung Cancer API is running 🚀"}

# @app.post("/predict")
# def predict(data: InputData):
#     try:

#         input_array = np.array([[ 
#             int(data.smoking),
#             int(data.yellow_fingers),
#             int(data.anxiety),
#             int(data.peer_pressure),
#             int(data.chronic_disease),
#             int(data.fatigue),
#             int(data.allergy),
#             int(data.wheezing),
#             int(data.alcohol_consuming),
#             int(data.coughing),
#             int(data.shortness_of_breath),
#             int(data.swallowing_difficulty),
#             int(data.chest_pain)
#             ]])
        
#         print("Array:", input_array)

#         prediction = model.predict(input_array)[0]

#         probability = None
#         if hasattr(model, "predict_proba"):
#             probability = model.predict_proba(input_array)[0][1]

#         result = "Cancer Detected" if prediction == 1 else "No Cancer"

#         return {
#             "prediction": int(prediction),
#             "probability": float(probability) if probability else None,
#             "result": result
#         }

#     except Exception as e:
#         print("❌ ERROR:", str(e))
#         return {"error": str(e)}

@app.post("/predict")
def predict(data: InputData):
    try:

        input_array = np.array([[ 
            data.smoking,
            data.yellow_fingers,
            data.anxiety,
            data.peer_pressure,
            data.chronic_disease,
            data.fatigue,
            data.allergy,
            data.wheezing,
            data.alcohol_consuming,
            data.coughing,
            data.shortness_of_breath,
            data.swallowing_difficulty,
            data.chest_pain
        ]])

        print("Array:", input_array)

        # FIX HERE
        prediction = model.predict(input_array)
        prediction = float(prediction.squeeze())

        probability = None
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(input_array)
            probability = float(prob.squeeze()[1])

        result = "Cancer Detected" if prediction >= 0.5 else "No Cancer"

        return {
            "prediction": prediction,
            "probability": probability,
            "result": result
        }

    except Exception as e:
        print("❌ ERROR:", str(e))
        return {"error": str(e)}