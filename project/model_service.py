import os
import joblib
import shap
import pandas as pd
import numpy as np

class MissingColumnsError(Exception):
    def __init__(self, missing_cols):
        self.missing_cols = missing_cols
        super().__init__(f"Missing columns: {missing_cols}")

class ModelService:
    def __init__(self, model_path="bankruptcy_model.pkl", features_path="feature_names.pkl"):
        # Resolve absolute paths based on this script's directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        m_path = os.path.join(current_dir, model_path)
        if not os.path.exists(m_path):
            m_path = os.path.join(os.path.dirname(current_dir), model_path)
            
        f_path = os.path.join(current_dir, features_path)
        if not os.path.exists(f_path):
            f_path = os.path.join(os.path.dirname(current_dir), features_path)
            
        if not os.path.exists(m_path) or not os.path.exists(f_path):
            raise FileNotFoundError(
                f"Model or features file not found. Paths checked: {m_path}, {f_path}"
            )
            
        self.model = joblib.load(m_path)
        self.feature_names = joblib.load(f_path)
        self.explainer = shap.TreeExplainer(self.model)

    def validate_and_prepare_df(self, df: pd.DataFrame) -> pd.DataFrame:
        missing_cols = [col for col in self.feature_names if col not in df.columns]
        if missing_cols:
            raise MissingColumnsError(missing_cols)
        return df[self.feature_names]

    def predict_risk(self, df: pd.DataFrame) -> dict:
        prepared_df = self.validate_and_prepare_df(df)
        
        # Predict class probabilities on the first sample
        pred = self.model.predict_proba(prepared_df)
        risk_score = float(pred[0][1] * 100)
        
        # Determine risk level
        if risk_score < 30.0:
            risk_level = "Low Risk"
        elif risk_score < 70.0:
            risk_level = "Medium Risk"
        else:
            risk_level = "High Risk"
            
        # Calculate SHAP values
        shap_values = self.explainer.shap_values(prepared_df)
        
        # Handle case where shap_values is a list of arrays (e.g. for classification model)
        if isinstance(shap_values, list):
            # If shape is (2, n_samples, n_features), shap_values[1] corresponds to class 1
            # In original code: shap_values[0] was used. Let's follow that but be safe
            raw_shap = shap_values[0]
        else:
            raw_shap = shap_values
            
        # Get SHAP values for the first sample
        sample_shap = raw_shap[0] if len(raw_shap.shape) > 1 else raw_shap

        # Map features to absolute SHAP values for importance
        feature_importance = pd.DataFrame({
            "Feature": prepared_df.columns,
            "SHAP Value": np.abs(sample_shap)
        })
        
        feature_importance = feature_importance.sort_values(
            by="SHAP Value",
            ascending=False
        )
        
        top_features = feature_importance.head(5).to_dict(orient="records")
        
        return {
            "risk_score": risk_score,
            "risk_level": risk_level,
            "top_features": top_features,
            "prepared_data": prepared_df.head(1).to_dict(orient="records")[0]
        }
