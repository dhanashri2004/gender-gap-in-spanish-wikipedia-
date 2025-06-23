import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_option_menu import option_menu

st.set_page_config(page_title="Gender Predictor", layout="centered")

# === Load model and encoder ===
try:
    model = joblib.load('model.pkl')
    c_api_encoder = joblib.load('c_api_encoder.pkl')
except Exception as e:
    st.error(f"⚠️ Failed to load model or encoder: {e}")
    st.stop()

# === Load dataset if available ===
# === Load dataset if available ===
try:
    df = pd.read_csv("data.csv")  # <-- this line loads your dataset
except Exception as e:
    st.warning(" Dataset not loaded. You can still use prediction.")
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
    st.write(
        """
        This app predicts a user's gender based on behavior and interaction features.
        
        📊 Explore data  
        📈 Visualize trends  
        🎯 Predict gender using our ML model  
        
        Use the sidebar to get started.
        """
    )

# === Data Preview Page ===
elif selected == "Data Preview":
    st.title("📋 Dataset Preview")
    if df is not None:
        st.dataframe(df.head(20))
        st.write(f"Shape of dataset: `{df.shape}`")
    else:
        st.warning("⚠️ Dataset not found or could not be loaded.")

# === Visualization Page ===
elif selected == "Visualization":
    st.title("📈 Data Visualization")
    if df is not None:
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        if numeric_cols:
            col = st.selectbox("Choose a numeric column to visualize", numeric_cols)
            fig, ax = plt.subplots()
            sns.histplot(df[col], kde=True, ax=ax)
            ax.set_title(f"Distribution of {col}")
            st.pyplot(fig)
        else:
            st.warning("⚠️ No numeric columns to plot.")
    else:
        st.warning("⚠️ Dataset not loaded.")

# === Predict Gender Page ===
elif selected == "Predict Gender":
    st.title("🎯 Predict Gender")

    try:
        # Input form
        with st.form("prediction_form"):
            api_input = st.selectbox("API", c_api_encoder.classes_)
            clicks = st.number_input("Number of Clicks", min_value=0, step=1)
            time_spent = st.number_input("Time Spent (seconds)", min_value=0.0, step=1.0)
            submit_btn = st.form_submit_button("Predict")

        if submit_btn:
            encoded_api = c_api_encoder.transform([api_input])[0]
            input_data = pd.DataFrame(
                [[encoded_api, clicks, time_spent]],
                columns=["API", "Clicks", "Time"]
            )
            st.write("📤 Input data:")
            st.dataframe(input_data)

            prediction = model.predict(input_data)[0]
            st.success(f"🧠 Predicted Gender: `{prediction}`")
    except Exception as e:
        st.error(f"⚠️ Prediction failed: {e}")
