import os
import pytest
import pandas as pd
from project.model_service import ModelService, MissingColumnsError

@pytest.fixture
def model_service():
    # ModelService searches relative to its own folder, so it should load successfully
    return ModelService()

@pytest.fixture
def sample_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sample_path = os.path.join(os.path.dirname(current_dir), "sample_company.csv")
    if not os.path.exists(sample_path):
        # Fallback to root workspace folder
        sample_path = os.path.join(os.path.dirname(os.path.dirname(current_dir)), "sample_company.csv")
    return pd.read_csv(sample_path)

def test_model_service_init(model_service):
    assert model_service.model is not None
    assert len(model_service.feature_names) > 0
    assert model_service.explainer is not None

def test_validate_and_prepare_df_success(model_service, sample_data):
    prepared_df = model_service.validate_and_prepare_df(sample_data)
    assert len(prepared_df.columns) == len(model_service.feature_names)
    assert list(prepared_df.columns) == list(model_service.feature_names)

def test_validate_and_prepare_df_missing_columns(model_service, sample_data):
    # Drop one required column
    bad_data = sample_data.drop(columns=[model_service.feature_names[0]])
    with pytest.raises(MissingColumnsError) as exc_info:
        model_service.validate_and_prepare_df(bad_data)
    assert model_service.feature_names[0] in exc_info.value.missing_cols

def test_predict_risk_success(model_service, sample_data):
    result = model_service.predict_risk(sample_data)
    
    assert "risk_score" in result
    assert "risk_level" in result
    assert "top_features" in result
    assert "prepared_data" in result
    
    assert isinstance(result["risk_score"], float)
    assert result["risk_level"] in ["Low Risk", "Medium Risk", "High Risk"]
    assert len(result["top_features"]) == 5
    assert len(result["prepared_data"]) == len(model_service.feature_names)
