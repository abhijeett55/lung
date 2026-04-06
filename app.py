import os
import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

model = joblib.load(os.path.join(os.path.dirname(__file__), "lung_model.pkl"))
print(type(model))

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
        print("Incoming:", data)

        input_array = np.array([[ 
            int(data.smoking),
            int(data.yellow_fingers),
            int(data.anxiety),
            int(data.peer_pressure),
            int(data.chronic_disease),
            int(data.fatigue),
            int(data.allergy),
            int(data.wheezing),
            int(data.alcohol_consuming),
            int(data.coughing),
            int(data.shortness_of_breath),
            int(data.swallowing_difficulty),
            int(data.chest_pain)
            ]])

        print("Array:", input_array)

        prediction = model.predict(input_array)
        prediction = float(prediction.flatten()[0])

        
        if prediction >= 1.5:
            result = "Cancer Detected"
        else:
            result = "No Cancer"

        return {
            "prediction": prediction,
            "result": result
        }

    except Exception as e:
        print("ERROR:", str(e))
        return {"error": str(e)}
