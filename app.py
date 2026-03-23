import pickle
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

model = pickle.load(open("lung_model.pkl", "rb"))

app = FastAPI()

class InputData(BaseModel):
    gender: int
    age: int
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
    input_array = np.array([[
        data.gender,
        data.age,
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

    prediction = model.predict(input_array)[0]

    return {
        "result": "Cancer Detected" if prediction == 1 else "No Cancer"
    }
