# api.py - Main FastAPI application
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import numpy as np
from typing import Dict, List
import uvicorn

# Load the trained model and preprocessing pipeline
try:
    model = joblib.load('networkAnalyser.pkl')
    print("Model loaded successfully!")
except FileNotFoundError:
    print("Model not found! Please train and save the model first.")
    model = None

app = FastAPI(
    title="Network Intrusion Detection API",
    description="API for predicting network intrusions using machine learning",
    version="1.0.0"
)


# Define the input schema based on UNSW-NB15 dataset features
class NetworkTraffic(BaseModel):
    # Network flow features (example - adjust based on your actual features)
    dur: float = 0.0
    proto: str = "tcp"
    service: str = "http"
    state: str = "FIN"
    spkts: int = 0
    dpkts: int = 0
    sbytes: int = 0
    dbytes: int = 0
    rate: float = 0.0
    sttl: int = 64
    dttl: int = 64
    sload: float = 0.0
    dload: float = 0.0
    sloss: int = 0
    dloss: int = 0
    sinpkt: float = 0.0
    dinpkt: float = 0.0
    sjit: float = 0.0
    djit: float = 0.0
    swin: int = 0
    stcpb: int = 0
    dtcpb: int = 0
    dwin: int = 0
    tcprtt: float = 0.0
    synack: float = 0.0
    ackdat: float = 0.0
    smean: float = 0.0
    dmean: float = 0.0
    trans_depth: int = 0
    response_body_len: int = 0
    ct_srv_src: int = 0
    ct_state_ttl: int = 0
    ct_dst_ltm: int = 0
    ct_src_dport_ltm: int = 0
    ct_dst_sport_ltm: int = 0
    ct_dst_src_ltm: int = 0
    is_ftp_login: int = 0
    ct_ftp_cmd: int = 0
    ct_flw_http_mthd: int = 0
    ct_src_ltm: int = 0
    ct_srv_dst: int = 0
    is_sm_ips_ports: int = 0


class PredictionResponse(BaseModel):
    prediction: int
    prediction_label: str
    confidence: float
    probabilities: Dict[str, float]


@app.get("/")
async def root():
    return {
        "message": "Network Intrusion Detection API",
        "status": "running",
        "model_loaded": model is not None
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_status": "loaded" if model else "not_loaded"
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_intrusion(traffic: NetworkTraffic):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        # Convert input to DataFrame
        input_data = pd.DataFrame([traffic.dict()])

        # Make prediction
        prediction = model.predict(input_data)[0]
        prediction_proba = model.predict_proba(input_data)[0]

        # Get confidence (max probability)
        confidence = float(max(prediction_proba))

        # Create probability dictionary
        classes = model.classes_ if hasattr(model, 'classes_') else ['Normal', 'Attack']
        probabilities = {
            str(classes[i]): float(prediction_proba[i])
            for i in range(len(prediction_proba))
        }

        # Convert prediction to label
        prediction_label = "Attack" if prediction == 1 else "Normal"

        return PredictionResponse(
            prediction=int(prediction),
            prediction_label=prediction_label,
            confidence=confidence,
            probabilities=probabilities
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")


@app.post("/predict_batch")
async def predict_batch(traffic_list: List[NetworkTraffic]):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    if len(traffic_list) > 100:
        raise HTTPException(status_code=400, detail="Batch size too large (max 100)")

    try:
        # Convert list to DataFrame
        input_data = pd.DataFrame([traffic.dict() for traffic in traffic_list])

        # Make predictions
        predictions = model.predict(input_data)
        predictions_proba = model.predict_proba(input_data)

        results = []
        classes = model.classes_ if hasattr(model, 'classes_') else ['Normal', 'Attack']

        for i, (pred, proba) in enumerate(zip(predictions, predictions_proba)):
            confidence = float(max(proba))
            probabilities = {
                str(classes[j]): float(proba[j])
                for j in range(len(proba))
            }
            prediction_label = "Attack" if pred == 1 else "Normal"

            results.append({
                "index": i,
                "prediction": int(pred),
                "prediction_label": prediction_label,
                "confidence": confidence,
                "probabilities": probabilities
            })

        return {"predictions": results, "count": len(results)}

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Batch prediction error: {str(e)}")


@app.get("/model_info")
async def get_model_info():
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        # Get model information
        model_type = type(model).__name__

        # Try to get additional info if it's a pipeline
        if hasattr(model, 'named_steps'):
            steps = list(model.named_steps.keys())
            classifier_type = type(model.named_steps.get('classifier', model)).__name__
        else:
            steps = ["direct_model"]
            classifier_type = model_type

        return {
            "model_type": model_type,
            "classifier_type": classifier_type,
            "pipeline_steps": steps,
            "has_predict_proba": hasattr(model, 'predict_proba')
        }
    except Exception as e:
        return {"error": f"Could not get model info: {str(e)}"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)