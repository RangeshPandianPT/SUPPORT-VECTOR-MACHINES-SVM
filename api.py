import io
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import shap
from sklearn.ensemble import RandomForestClassifier

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

# Load data and initialize SHAP explainer
try:
    df = pd.read_csv('breast-cancer.csv')
    df.drop(columns=['id'], inplace=True, errors='ignore')
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
    X = df.drop('diagnosis', axis=1)
    y = df['diagnosis']
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X, y)
    explainer = shap.TreeExplainer(rf_model)
except Exception as e:
    explainer = None
    print(f"Warning: Could not initialize SHAP explainer. Error: {e}")

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

@app.post("/batch-predict")
async def batch_predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Model is not loaded.")
    
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")
        
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        df_features = df.drop(columns=['id', 'diagnosis'], errors='ignore')
        
        predictions = model.predict(df_features)
        probabilities = model.predict_proba(df_features)
        
        results = []
        for i, pred in enumerate(predictions):
            results.append({
                "index": i,
                "prediction": "Malignant" if pred == 1 else "Benign",
                "confidence": f"{float(probabilities[i][pred]) * 100:.2f}%"
            })
            
        return {"batch_results": results}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Batch prediction error: {str(e)}")

@app.get("/metrics/confusion-matrix")
def get_confusion_matrix():
    import os
    if os.path.exists("confusion_matrix.png"):
        return FileResponse("confusion_matrix.png", media_type="image/png")
    raise HTTPException(status_code=404, detail="Confusion matrix not found")

@app.get("/metrics/pca-plot")
def get_pca_plot():
    import os
    if os.path.exists("svm_rbf_pca.png"):
        return FileResponse("svm_rbf_pca.png", media_type="image/png")
    raise HTTPException(status_code=404, detail="PCA plot not found")

@app.post("/explain")
def explain(features: TumorFeatures):
    if explainer is None:
        raise HTTPException(status_code=500, detail="Explainer is not loaded.")
    
    data = pd.DataFrame([features.dict()])
    shap_values = explainer.shap_values(data)
    
    if isinstance(shap_values, list):
        instance_shap = shap_values[1][0] 
    else:
        instance_shap = shap_values[0]
        
    importance = {feature: float(val) for feature, val in zip(data.columns, instance_shap)}
    sorted_importance = dict(sorted(importance.items(), key=lambda item: abs(item[1]), reverse=True))
    
    return {"feature_importance": sorted_importance}

