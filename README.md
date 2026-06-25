# Bankruptcy Risk Analyzer

An AI-powered financial risk assessment system that predicts the probability of corporate bankruptcy using machine learning and provides explainable insights through SHAP visualizations.

## Features

* Predicts bankruptcy risk from company financial statements using XGBoost.
* Handles class imbalance using SMOTE.
* Provides prediction confidence and risk level.
* Generates SHAP-based feature importance for explainability.
* Stores prediction history in SQLite.
* Interactive Streamlit dashboard for CSV upload and visualization.
* FastAPI backend exposing prediction APIs.

## Tech Stack

* Python
* XGBoost
* FastAPI
* Streamlit
* SHAP
* Scikit-learn
* SQLite
* Pandas
* NumPy

## Model Performance

| Metric    | Score  |
| --------- | ------ |
| Accuracy  | 96.52% |
| ROC-AUC   | 0.928  |
| Precision | 0.400  |
| Recall    | 0.452  |
| F1 Score  | 0.424  |

## Project Structure

```
project/
├── app.py                # Streamlit frontend
├── main.py               # FastAPI backend
├── bankruptcy_model.pkl
├── predictions.db
├── requirements.txt
├── utils/
└── assets/
```

## Installation

```bash
git clone https://github.com/<your-username>/Bankruptcy-Risk-Analyzer.git
cd Bankruptcy-Risk-Analyzer

pip install -r requirements.txt
```

## Run the Backend

```bash
uvicorn main:app --reload
```

## Run the Frontend

```bash
streamlit run app.py
```

## Future Improvements

* Docker deployment
* Cloud deployment on AWS
* User authentication
* Batch prediction support
* PDF financial risk reports

## Author

**Palak Shrivastava**
