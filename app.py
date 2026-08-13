import streamlit as st
import pandas as pd
from pathlib import Path

# --------------------------------------------------
# LOAD CUSTOM CSS
# --------------------------------------------------

css_path = Path("assets/style.css")

if css_path.exists():
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(
            f"<style>{f.read()}</style>",
            unsafe_allow_html=True
        )
        
from utils.load_model import load_trained_model
from utils.predictor import predict_rul
from utils.helpers import (
    get_health_status,
    get_status_color,
    format_rul,
)

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Bearing Lifetime Prediction",
    page_icon="⚙️",
    layout="wide",
)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

model = load_trained_model()

# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "history" not in st.session_state:
    st.session_state.history = []

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.title("⚙️ Project Information")

    st.markdown("### 📌 Project")
    st.write("Bearing Lifetime Prediction")

    st.markdown("### 🤖 Model")
    st.write("Random Forest Regressor")

    st.markdown("### 📂 Dataset")
    st.write("IMS Bearing Dataset")

    st.markdown("### 📊 Features")

    st.markdown("""
- Mean
- Standard Deviation
- RMS
- Kurtosis
- Skewness
- Peak-to-Peak
""")

    st.markdown("### 👩‍💻 Developed By")
    st.write("Aayushi Patel")

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("⚙️ Bearing Lifetime Prediction Dashboard")

st.write("""
Predict the **Remaining Useful Life (RUL)** of industrial bearings
using Machine Learning based on vibration signal analysis.
""")

st.divider()

# --------------------------------------------------
# MODEL PERFORMANCE
# --------------------------------------------------

st.subheader("📈 Model Performance")

c1, c2, c3 = st.columns(3)

with c1:
    st.metric("MAE", "791.29")

with c2:
    st.metric("RMSE", "1160.80")

with c3:
    st.metric("R² Score", "0.6090")

st.divider()

# --------------------------------------------------
# INPUT SECTION
# --------------------------------------------------

st.subheader("📥 Enter Vibration Features")

left, right = st.columns(2)

with left:
    mean = st.number_input(
        "Mean",
        value=-0.094593,
        format="%.6f"
    )

    std = st.number_input(
        "Standard Deviation",
        value=0.081122,
        format="%.6f"
    )

    rms = st.number_input(
        "RMS",
        value=0.124614,
        format="%.6f"
    )

with right:
    kurtosis = st.number_input(
        "Kurtosis",
        value=1.069163,
        format="%.6f"
    )

    skewness = st.number_input(
        "Skewness",
        value=-0.029993,
        format="%.6f"
    )

    peak_to_peak = st.number_input(
        "Peak-to-Peak",
        value=1.108000,
        format="%.6f"
    )

st.divider()

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if st.button("🚀 Predict Remaining Useful Life", use_container_width=True):

    prediction = predict_rul(
        model=model,
        mean=mean,
        std=std,
        rms=rms,
        kurtosis=kurtosis,
        skewness=skewness,
        peak_to_peak=peak_to_peak
    )

    status = get_health_status(prediction)
    color = get_status_color(prediction)

    # Save prediction history
    st.session_state.history.append({
        "Mean": mean,
        "Std": std,
        "RMS": rms,
        "Kurtosis": kurtosis,
        "Skewness": skewness,
        "Peak_to_Peak": peak_to_peak,
        "Predicted_RUL": round(prediction, 2),
        "Status": status
    })

    st.success("✅ Prediction Completed Successfully")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            label="Predicted Remaining Useful Life",
            value=format_rul(prediction)
        )

    with col2:
        st.markdown(
            f"""
            <div style="
                background-color:{color};
                padding:20px;
                border-radius:12px;
                text-align:center;
                color:white;
                font-size:22px;
                font-weight:bold;">
                {status}
            </div>
            """,
            unsafe_allow_html=True,
        )

st.divider()

# --------------------------------------------------
# FEATURE IMPORTANCE
# --------------------------------------------------

feature_path = Path("outputs/feature_importance.png")

if feature_path.exists():

    st.subheader("📊 Feature Importance")

    st.image(
        str(feature_path),
        use_container_width=True
    )

# --------------------------------------------------
# CORRELATION HEATMAP
# --------------------------------------------------

heatmap_path = Path("outputs/correlation_heatmap.png")

if heatmap_path.exists():

    st.subheader("🔥 Correlation Heatmap")

    st.image(
        str(heatmap_path),
        use_container_width=True
    )

# --------------------------------------------------
# ACTUAL VS PREDICTED GRAPH
# --------------------------------------------------

prediction_graph = Path("outputs/actual_vs_predicted.png")

if prediction_graph.exists():

    st.subheader("📉 Actual vs Predicted")

    st.image(
        str(prediction_graph),
        use_container_width=True
    )

st.divider()

# --------------------------------------------------
# PREDICTION HISTORY
# --------------------------------------------------

st.subheader("📋 Prediction History")

if len(st.session_state.history) > 0:

    history_df = pd.DataFrame(st.session_state.history)

    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True
    )

    csv = history_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📥 Download Prediction History",
        data=csv,
        file_name="prediction_history.csv",
        mime="text/csv",
        use_container_width=True
    )

else:

    st.info("No predictions have been made yet.")

st.divider()

# --------------------------------------------------
# ABOUT PROJECT
# --------------------------------------------------

st.subheader("📖 About This Project")

st.markdown("""
This application predicts the **Remaining Useful Life (RUL)** of industrial
bearings using Machine Learning.

The model is trained on the **IMS Bearing Dataset**, where statistical
features are extracted from vibration signals to estimate the remaining
lifetime of bearings before failure.

The objective of this project is to demonstrate how predictive maintenance
can reduce unexpected machine failures and improve industrial reliability.
""")

st.divider()

# --------------------------------------------------
# TECHNOLOGY STACK
# --------------------------------------------------

st.subheader("🛠 Technology Stack")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
### Machine Learning
- Random Forest Regression
- Scikit-learn
- Pandas
- NumPy
""")

with col2:
    st.markdown("""
### Deployment
- Streamlit
- Joblib
- Matplotlib
- Python
""")

st.divider()

# --------------------------------------------------
# PROJECT FEATURES
# --------------------------------------------------

st.subheader("✨ Features")

st.markdown("""
✅ Predict Remaining Useful Life (RUL)

✅ Bearing Health Indicator

✅ Prediction History

✅ Download Prediction History

✅ Feature Importance Visualization

✅ Correlation Heatmap

✅ Machine Learning Dashboard

✅ Industrial Predictive Maintenance
""")

st.divider()

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown(
    """
    <div style="text-align:center; color:gray; padding:20px;">
        Developed by <b>Aayushi Patel</b><br>
        Bearing Lifetime Prediction using Machine Learning
    </div>
    """,
    unsafe_allow_html=True,
)