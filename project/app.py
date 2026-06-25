import os
import streamlit as st
import pandas as pd
import requests

# API configuration - uses environment variable or defaults to localhost
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.title("AI-Powered Bankruptcy Risk Analyzer")

uploaded_file = st.file_uploader(
    "Upload CSV",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    st.write("Uploaded Data")
    st.dataframe(df.head())
    
    if st.button("Predict"):
        payload = df.to_dict(orient="records")
        
        try:
            response = requests.post(f"{API_URL}/predict", json=payload)
            
            if response.status_code == 400:
                # Handle validation error (e.g. missing columns)
                err_detail = response.json().get("detail", "")
                st.error(err_detail)
                st.stop()
            elif response.status_code != 200:
                st.error(f"Prediction failed with status code {response.status_code}: {response.text}")
                st.stop()
                
            result = response.json()
            risk = result["risk_score"]
            risk_level = result["risk_level"]
            top_features = pd.DataFrame(result["top_features"])
            
            st.metric(
                "Bankruptcy Risk",
                f"{risk:.2f}%"
            )
            
            st.subheader("Prediction Summary")
            st.write(
                f"""
                The model predicts a bankruptcy risk of
                {risk:.2f}%.

                The most influential financial indicators
                are shown below.
                """
            )
            
            if risk_level == "Low Risk":
                st.success("Low Risk")
            elif risk_level == "Medium Risk":
                st.warning("Medium Risk")
            else:
                st.error("High Risk")
                
            st.subheader("Top Risk Factors")
            
            chart_data = top_features.set_index("Feature")
            st.bar_chart(chart_data)
            
            top_feature_names = top_features["Feature"].tolist()
            feature_text = "\n".join(
                [f"- {f}" for f in top_feature_names]
            )
            
            st.subheader("Financial Risk Report")
            st.write(f"""
            ### Executive Summary

            Predicted Bankruptcy Risk: **{risk:.2f}%**

            ### Key Financial Drivers

            {feature_text}

            ### Recommendation

            Management should closely monitor the above indicators and take corrective measures where necessary.
            """)
            
        except requests.exceptions.ConnectionError:
            st.error(f"Could not connect to the Backend API at {API_URL}. Please verify the backend is running.")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")

st.subheader("Prediction History")

if st.button("Show History"):
    try:
        response = requests.get(f"{API_URL}/history")
        
        if response.status_code == 200:
            history = pd.DataFrame(response.json())
            st.dataframe(history)
        else:
            st.error(f"Failed to fetch history: {response.text}")
            
    except requests.exceptions.ConnectionError:
        st.error(f"Could not connect to the Backend API at {API_URL}. Please verify the backend is running.")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
