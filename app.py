import os
import joblib
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
    return {"message": "API running 🚀"}


@app.post("/predict")
def predict(data: InputData):
    try:
        
        input_df = pd.DataFrame([{
            "SMOKING": data.smoking,
            "YELLOW_FINGERS": data.yellow_fingers,
            "ANXIETY": data.anxiety,
            "PEER_PRESSURE": data.peer_pressure,
            "CHRONIC DISEASE": data.chronic_disease,
            "FATIGUE ": data.fatigue,
            "ALLERGY ": data.allergy,
            "WHEEZING": data.wheezing,
            "ALCOHOL CONSUMING": data.alcohol_consuming,
            "COUGHING": data.coughing,
            "SHORTNESS OF BREATH": data.shortness_of_breath,
            "SWALLOWING DIFFICULTY": data.swallowing_difficulty,
            "CHEST PAIN": data.chest_pain
        }])

        print("📥 Input:\n", input_df)

        
        prediction = model.predict(input_df)
        prediction_value = prediction.item()

        print("🧠 Raw prediction:", prediction_value)

        
        result = "Cancer Detected" if prediction_value >= 1.2 else "No Cancer"

        return {
            "prediction": float(prediction_value),
            "result": result
        }

    except Exception as e:
        print("❌ ERROR:", str(e))
        return {"error": str(e)}