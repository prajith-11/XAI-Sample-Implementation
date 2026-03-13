from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware  # <--- Make sure this is imported
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import shap

app = FastAPI()

# THIS BLOCK IS ESSENTIAL
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows the HTML file to talk to the API
    allow_credentials=True,
    allow_methods=["*"],  # Allows POST, OPTIONS, etc.
    allow_headers=["*"],
)

# 1. Define the data structure (Matches your generation code)
class Student(BaseModel):
    Attendance: int
    Internal_Marks: int
    Study_Hours: int
    Assignment_Score: int

# 2. Load the Brains
model = joblib.load('student_model.pkl')
explainer = joblib.load('shap_explainer.pkl')
feature_names = joblib.load('features.pkl')

@app.post("/predict")
async def predict_student(student: Student):
    # Convert Pydantic object to Dictionary, then to DataFrame
    data_dict = student.dict()
    input_df = pd.DataFrame([data_dict])
    
    # 3. Prediction
    pred = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0]
    result = "Pass" if pred == 1 else "Fail"
    confidence = max(prob) * 100

    # 4. SHAP Logic
    sv = explainer(input_df)
    # Use index 1 if your model is binary (0=Fail, 1=Pass)
    impacts = sv.values[0, :, 1] 
    top_idx = np.abs(impacts).argmax()
    
    reason_feature = feature_names[top_idx]
    reason_value = data_dict[reason_feature]
    impact_score = impacts[top_idx]

    direction = "pushed" if impact_score > 0 else "pulled"
    side = "towards" if impact_score > 0 else "away from"

    ai_insight = (f"Predicted {result} because {reason_feature} ({reason_value}) "
                  f"{direction} the prediction {side} Pass.")

    return {
        "prediction": result,
        "confidence": f"{confidence:.2f}%",
        "insight": ai_insight
    }