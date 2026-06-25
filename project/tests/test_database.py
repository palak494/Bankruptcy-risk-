import pytest
from project.database import DatabaseHandler

def test_database_init():
    # Use in-memory database to avoid touching disk
    db = DatabaseHandler(":memory:")
    conn = db.get_connection()
    tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    assert ("prediction_history",) in tables
    conn.close()

def test_insert_and_get_history():
    db = DatabaseHandler(":memory:")
    db.insert_prediction(
        risk_score=45.5,
        risk_level="Medium Risk",
        top_features=["Feature A", "Feature B", "Feature C"]
    )
    
    history_df = db.get_history()
    assert len(history_df) == 1
    row = history_df.iloc[0]
    assert row["risk_score"] == 45.5
    assert row["risk_level"] == "Medium Risk"
    assert row["top_feature_1"] == "Feature A"
    assert row["top_feature_2"] == "Feature B"
    assert row["top_feature_3"] == "Feature C"

def test_insert_with_missing_features():
    db = DatabaseHandler(":memory:")
    # Insert with only 1 feature to test automatic padding
    db.insert_prediction(
        risk_score=10.0,
        risk_level="Low Risk",
        top_features=["Single Feature"]
    )
    history_df = db.get_history()
    assert len(history_df) == 1
    row = history_df.iloc[0]
    assert row["top_feature_1"] == "Single Feature"
    assert row["top_feature_2"] is None
    assert row["top_feature_3"] is None
