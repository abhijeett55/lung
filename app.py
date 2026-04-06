import os
import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI app
app = FastAPI()

# Load model once at startup
MODEL_PATH = os.path.join(os.path.dirname(__file__), "lung_model.pkl")
model = joblib.load(MODEL_PATH)
print("✅ Model loaded:", type(model))


# Enable CORS (for frontend connection)
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


# Health check route
@app.get("/")
def home():
    return {"message": "Lung Cancer Prediction API is running 🚀"}


# Prediction route
@app.post("/predict")
def predict(data: InputData):
    try:
        # Convert input to array (order MUST match training)
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

        print("📥 Received Input:", input_array)

        # Predict
        prediction = model.predict(input_array)[0]

        # Probability (only if classifier supports it)
        probability = None
        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(input_array)[0][1]

        # Result label
        result = "Cancer Detected" if prediction == 1 else "No Cancer"

        return {
            "prediction": int(prediction),
            "probability": float(probability) if probability is not None else None,
            "result": result
        }

    except Exception as e:
        print("❌ ERROR:", str(e))
        return {
            "error": str(e)
        }