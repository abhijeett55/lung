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


@app.post("/predict")
def predict(data: InputData):
    try:

        input_df = pd.DataFrame([{
            "SMOKING": data.smoking,
            "YELLOW_FINGERS": data.yellow_fingers,
            "ANXIETY": data.anxiety,
            "PEER_PRESSURE": data.peer_pressure,
            "CHRONIC DISEASE": data.chronic_disease,
            "FATIGUE": data.fatigue,
            "ALLERGY": data.allergy,
            "WHEEZING": data.wheezing,
            "ALCOHOL CONSUMING": data.alcohol_consuming,
            "COUGHING": data.coughing,
            "SHORTNESS OF BREATH": data.shortness_of_breath,
            "SWALLOWING DIFFICULTY": data.swallowing_difficulty,
            "CHEST PAIN": data.chest_pain
        }])

        # clean columns
        input_df.columns = input_df.columns.str.strip()
        model_features = [col.strip() for col in model.feature_names_in_]

        input_df = input_df.reindex(columns=model_features, fill_value=0)

        prediction = model.predict(input_df)[0]

        probability = None
        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(input_df)[0][1]

        result = "Cancer Detected" if prediction == 1 else "No Cancer"

        return {
            "prediction": int(prediction),
            "probability": float(probability) if probability else None,
            "result": result
        }

    except Exception as e:
        print("❌ ERROR:", str(e))
        return {"error": str(e)}