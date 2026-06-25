from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any
import pandas as pd

from project.database import DatabaseHandler
from project.model_service import ModelService, MissingColumnsError

app = FastAPI(title="Company Bankruptcy API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instantiate handler and service
# The absolute paths are automatically resolved inside classes
db = DatabaseHandler("predictions.db")
model_service = ModelService("bankruptcy_model.pkl", "feature_names.pkl")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(payload: List[Dict[str, Any]]):
    if not payload:
        raise HTTPException(status_code=400, detail="Data payload cannot be empty.")
    
    try:
        df = pd.DataFrame(payload)
        result = model_service.predict_risk(df)
        
        # Log run in database
        risk_score = result["risk_score"]
        risk_level = result["risk_level"]
        top_features_list = [item["Feature"] for item in result["top_features"]]
        
        db.insert_prediction(
            risk_score=risk_score,
            risk_level=risk_level,
            top_features=top_features_list[:3]
        )
        
        return result
        
    except MissingColumnsError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/history")
def history():
    try:
        history_df = db.get_history()
        # Convert NaN values to None for proper JSON serialization
        history_df = history_df.where(pd.notnull(history_df), None)
        return history_df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch history: {str(e)}")
