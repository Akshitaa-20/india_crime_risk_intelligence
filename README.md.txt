# 🇮🇳 India Crime Risk Intelligence

An end-to-end AIML project that analyzes, models, and forecasts crimes against women in India using historical data (2001–2022).  
The project combines data analytics, machine learning, backend services, and an interactive dashboard.

---

## Problem Statement

Crimes against women are a critical social issue in India.  
Understanding long-term trends and predicting future patterns can help policymakers, researchers, and analysts make data-driven decisions.

This project aims to:
- Analyze historical crime trends
- Engineer meaningful time-series features
- Build predictive machine learning models
- Forecast future crime counts
- Present insights through an interactive dashboard

---

## Key Features

- Trend analysis of crimes against women (2001–2022)
- Time-series feature engineering (lag features, rolling averages, growth rate)
- Machine learning-based crime prediction
- Multi-year crime forecasting
- Interactive Streamlit dashboard
- Clean frontend–backend separation

---

---

## Tech Stack

- **Programming:** Python
- **Data Analysis:** Pandas, NumPy
- **Machine Learning:** Scikit-learn
- **Visualization:** Matplotlib
- **Dashboard:** Streamlit
- **Model Persistence:** Joblib

---

## Machine Learning Approach

- Framed the task as a **time-series regression problem**
- Used historical crime counts to predict future values
- Implemented:
  - Lag features (previous year dependencies)
  - Rolling averages (trend smoothing)
  - Year-on-year growth rate (momentum capture)
- Trained and evaluated:
  - Linear Regression (baseline)
  - Random Forest Regressor (final model)

---

## Results & Insights

- Long-term crime trend shows a general increase over years
- Certain crime categories consistently dominate statistics
- Lag-based features were the strongest predictors
- Forecast indicates continued rise in reported crimes in the near future

---

## How to Run the Project

### Install Dependencies
```bash
pip install streamlit pandas matplotlib scikit-learn joblib

##Live Demo
https://indiacrimeriskintelligence-hayapc2gyg6abjophvdlrn.streamlit.app/

