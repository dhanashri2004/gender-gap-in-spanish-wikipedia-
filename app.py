import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_option_menu import option_menu

st.set_page_config(page_title="Gender Predictor", layout="centered")

# === Load model and encoder ===
try:
    model = joblib.load("model.pkl")
    c_api_encoder = joblib.load("c_api_encoder.pkl")
except Exception as e:
    st.error(f"⚠️ Failed to load model or encoder: {e}")
    st.stop()

# === Load dataset ===
try:
    df = pd.read_csv("data.csv")
except Exception as e:
    st.warning("⚠️ Dataset not loaded. You can still use prediction.")
    df = None

# === Sidebar Menu ===
with st.sidebar:
    selected = option_menu(
        menu_title="Main Menu",
        options=["Home", "Data Preview", "Visualization", "Predict Gender"],
        icons=["house", "table", "bar-chart", "person-check"],
        menu_icon="cast",
        default_index=0,
    )

# === Home Page ===
if selected == "Home":
    st.title("👋 Welcome to the Gender Prediction App")
    st.write("""
        This app predicts a user's gender based on interaction data.

        📊 Explore Data  
        📈 Visualize Trends  
        🎯 Predict Gender  
        
        Use the sidebar to navigate.
    """)

# === Data Preview ===
elif selected == "Data Preview":
    st.title("📋 Dataset Preview")
    if df is not None:
        st.dataframe(df.head(20))
        st.write(f"Dataset shape: {df.shape}")
    else:
        st.warning("⚠️ Dataset not loaded.")

# === Visualization ===
elif selected == "Visualization":
    st.title("📈 Data Visualization")
    if df is not None:
        num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        if num_cols:
            col = st.selectbox("Choose numeric column to visualize", num_cols)
            fig, ax = plt.subplots()
            sns.histplot(df[col], kde=True, ax=ax)
            ax.set_title(f"Distribution of {col}")
            st.pyplot(fig)
        else:
            st.warning("⚠️ No numeric columns found.")
    else:
        st.warning("⚠️ Dataset not loaded.")

# === Predict Gender ===
elif selected == "Predict Gender":
    st.title("🎯 Predict Gender")

    with st.form("prediction_form"):
        # Encoded field
        api_input = st.selectbox("API", c_api_encoder.classes_)

        # Numeric fields
        clicks = st.number_input("Number of Clicks", min_value=0, step=1)
        time_spent = st.number_input("Time Spent (seconds)", min_value=0.0, step=1.0)
        n_act_days = st.number_input("Number of Active Days", min_value=0, step=1)

        # Binary feature inputs
        c_api = st.selectbox("C_api", ["Yes", "No"])
        c_man = st.selectbox("C_man", ["Yes", "No"])
        e_bpag = st.selectbox("E_Bpag", ["Yes", "No"])
        e_neds = st.selectbox("E_NEds", ["Yes", "No"])

        # Submit
        submit_btn = st.form_submit_button("Predict")

    if submit_btn:
        try:
            encoded_api = c_api_encoder.transform([api_input])[0]

            # Create input DataFrame matching model's expected structure
            input_data = pd.DataFrame([{
                "API": encoded_api,
                "Clicks": clicks,
                "Time": time_spent,
                "C_api": 1 if c_api == "Yes" else 0,
                "C_man": 1 if c_man == "Yes" else 0,
                "E_Bpag": 1 if e_bpag == "Yes" else 0,
                "E_NEds": 1 if e_neds == "Yes" else 0,
                "NActDays": n_act_days
            }])

            st.subheader("📥 Input Data")
            st.dataframe(input_data)

            # Predict
            prediction = model.predict(input_data)[0]
            st.success(f"🧠 Predicted Gender: `{prediction}`")

        except Exception as e:
            st.error(f"⚠️ Prediction failed: {e}")
