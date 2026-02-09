import streamlit as st
import matplotlib.pyplot as plt
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, "..", "backend"))

from predictor import (
    get_historical_data,
    get_feature_data,
    forecast_next_n_years
)

st.set_page_config(
    page_title="India Crime Risk Intelligence",
    layout="wide"
)

st.title("📊 India Crime Risk Intelligence Dashboard")
st.markdown("Full-stack AIML system for crime trend analysis and forecasting")

# ======================
# Historical Trend
# ======================
st.subheader("📈 Historical Crime Trend")

hist_df = get_historical_data()

fig, ax = plt.subplots(figsize=(8,4))
ax.plot(hist_df["Year"], hist_df["Crime_Count"], marker="o")
ax.set_xlabel("Year")
ax.set_ylabel("Crime Count")
ax.grid(True)

st.pyplot(fig)

# ======================
# Feature Explorer
# ======================
st.subheader("🧠 Feature Explorer")

feature_df = get_feature_data()
feature_cols = feature_df.drop(columns=["Year", "Crime_Count"]).columns

selected_feature = st.selectbox("Select Feature", feature_cols)

fig2, ax2 = plt.subplots(figsize=(8,4))
ax2.plot(feature_df["Year"], feature_df[selected_feature], marker="o")
ax2.set_title(selected_feature)
ax2.grid(True)

st.pyplot(fig2)

# ======================
# Forecast Section
# ======================
st.subheader("🔮 Crime Forecast")

years = st.slider("Select number of future years", 1, 5, 3)
forecast_df = forecast_next_n_years(years)

st.dataframe(forecast_df)

fig3, ax3 = plt.subplots(figsize=(8,4))
ax3.plot(hist_df["Year"], hist_df["Crime_Count"], label="Historical", marker="o")
ax3.plot(
    forecast_df["Year"],
    forecast_df["Predicted_Crime"],
    label="Forecast",
    marker="o",
    linestyle="--"
)
ax3.legend()
ax3.grid(True)

st.pyplot(fig3)

st.markdown("---")
st.markdown("Built with ❤️ using Python, Machine Learning & Streamlit")
