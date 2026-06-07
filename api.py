from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

app = FastAPI(title="Breast Cancer SVM Prediction API", 
              description="A REST API to predict whether a breast tumor is benign or malignant using a trained SVM model.",
              version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # For development, allow all
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Load the trained model
try:
    model = joblib.load('best_svm_model.pkl')
except Exception as e:
    model = None
    print(f"Warning: Could not load model. Error: {e}")

# Define the expected input payload using Pydantic
class TumorFeatures(BaseModel):
    radius_mean: float
    texture_mean: float
    perimeter_mean: float
    area_mean: float
    smoothness_mean: float
    compactness_mean: float
    concavity_mean: float
    concave_points_mean: float
    symmetry_mean: float
    fractal_dimension_mean: float
    radius_se: float
    texture_se: float
    perimeter_se: float
    area_se: float
    smoothness_se: float
    compactness_se: float
    concavity_se: float
    concave_points_se: float
    symmetry_se: float
    fractal_dimension_se: float
    radius_worst: float
    texture_worst: float
    perimeter_worst: float
    area_worst: float
    smoothness_worst: float
    compactness_worst: float
    concavity_worst: float
    concave_points_worst: float
    symmetry_worst: float
    fractal_dimension_worst: float

@app.get("/")
def read_root():
    return {"message": "Welcome to the Breast Cancer Prediction API. Use the /predict endpoint to get predictions."}

@app.post("/predict")
def predict(features: TumorFeatures):
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded. Please train the model first.")
    
    # Convert input to DataFrame (as the pipeline expects column names, though numpy array also works for just standard scaler usually)
    data = pd.DataFrame([features.dict()])
    
    try:
        # Get prediction and probabilities
        prediction = model.predict(data)[0]
        probabilities = model.predict_proba(data)[0]
        
        result = "Malignant" if prediction == 1 else "Benign"
        confidence = probabilities[prediction]
        
        return {
            "prediction": result,
            "confidence": f"{confidence * 100:.2f}%",
            "probabilities": {
                "Benign": f"{probabilities[0] * 100:.2f}%",
                "Malignant": f"{probabilities[1] * 100:.2f}%"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")
