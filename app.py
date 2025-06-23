import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_option_menu import option_menu  # ✅ Correct import

st.set_page_config(page_title="Gender Predictor", layout="centered")

# === Load model and encoder ===
try:
    model = joblib.load('model.pkl')
    c_api_encoder = joblib.load('c_api_encoder.pkl')
except Exception as e:
    st.error(f"⚠️ Failed to load model or encoder: {e}")
    st.stop()

# === Load dataset if available ===
try:
    df = pd.read_csv("data.csv")
except Exception:
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
        This app predicts a user's gender based on behavior and interaction features.

        📊 Explore data  
        📈 Visualize trends  
        🎯 Predict gender using our ML model  

        Use the sidebar to get started.
    """)

# === Data Preview Page ===
elif selected == "Data Preview":
    st.title("📋 Dataset Preview")
    if df is not None:
        st.dataframe(df.head(20))
        st.write(f"Shape: {df.shape}")
    else:
        st.warning("⚠️ Dataset not loaded.")

# === Visualization Page ===
elif selected == "Visualization":
    st.title("📈 Data Visualization")
    if df is not None:
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        if numeric_cols:
            col = st.selectbox("Choose a numeric column", numeric_cols)
            fig, ax = plt.subplots()
            sns.histplot(df[col], kde=True, ax=ax)
            ax.set_title(f"Distribution of {col}")
            st.pyplot(fig)
        else:
            st.warning("⚠️ No numeric columns available.")
    else:
        st.warning("⚠️ Dataset not loaded.")

# === Predict Gender Page ===
elif selected == "Predict Gender":
    st.title("🎯 Predict Gender")

    try:
        with st.form("prediction_form"):
            c_api = st.selectbox("C_api", c_api_encoder.classes_)
            c_man = st.number_input("C_man", min_value=0)
            e_bpag = st.number_input("E_Bpag", min_value=0)
            e_neds = st.number_input("E_NEds", min_value=0)
            nactdays = st.number_input("NActDays", min_value=0)
            submit_btn = st.form_submit_button("Predict")

        if submit_btn:
            encoded_api = c_api_encoder.transform([c_api])[0]
            input_data = pd.DataFrame(
                [[encoded_api, c_man, e_bpag, e_neds, nactdays]],
                columns=["C_api", "C_man", "E_Bpag", "E_NEds", "NActDays"]
            )
            st.write("📤 Input data:")
            st.dataframe(input_data)

            prediction = model.predict(input_data)[0]
            st.success(f"🧠 Predicted Gender: `{prediction}`")

    except Exception as e:
        st.error(f"⚠️ Prediction failed: {e}")
