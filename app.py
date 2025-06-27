import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_option_menu import option_menu

st.set_page_config(page_title="Gender Predictor", layout="centered")

# === Load model and encoders ===
try:
    model = joblib.load("model.pkl")
    c_api_encoder = joblib.load("c_api_encoder.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
except Exception as e:
    st.error(f"⚠️ Failed to load model or encoders: {e}")
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
        This app predicts a user's gender based on Wikipedia edit behavior.

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
        api_input = st.selectbox("C_api (User Language)", c_api_encoder.classes_)
        c_man = st.selectbox("C_man (Manual Curation)", ["0", "1", "2", "3"])
        e_neds = st.selectbox("E_NEds (Education Sessions)", ["0", "1", "2", "3"])
        e_bpag = st.selectbox("E_Bpag (Blog Pages)", ["0", "1", "2", "3"])

        neds = st.number_input("NEds (Number of Edits)", min_value=0, step=1)
        ndays = st.number_input("NDays (Total Days Active)", min_value=0, step=1)
        nactdays = st.number_input("NActDays (Active Days)", min_value=0, step=1)
        npages = st.number_input("NPages", min_value=0, step=1)
        npcreated = st.number_input("NPcreated (Pages Created)", min_value=0, step=1)
        nij = st.number_input("NIJ (Index)", min_value=0, step=1)
        weightij = st.number_input("WeightIJ", min_value=0.0, step=0.01)
        pages_women = st.number_input("pagesWomen", min_value=0, step=1)
        proj_women = st.number_input("wikiprojWomen", min_value=0, step=1)

        ns_user = st.number_input("ns_user", min_value=0, step=1)
        ns_wiki = st.number_input("ns_wikipedia", min_value=0, step=1)
        ns_talk = st.number_input("ns_talk", min_value=0, step=1)
        ns_userTalk = st.number_input("ns_userTalk", min_value=0, step=1)
        ns_content = st.number_input("ns_content", min_value=0, step=1)

        submit_btn = st.form_submit_button("Predict")
if submit_btn:
    try:
        encoded_api = c_api_encoder.transform([api_input])[0]

        input_data = pd.DataFrame([{
            "C_api": encoded_api,
            "C_man": int(c_man),
            "E_NEds": int(e_neds),
            "E_Bpag": int(e_bpag),
            "NEds": neds,
            "NDays": ndays,
            "NActDays": nactdays,
            "NPages": npages,
            "NPcreated": npcreated,
            "pagesWomen": pages_women,
            "wikiprojWomen": proj_women,
            "ns_user": ns_user,
            "ns_wikipedia": ns_wiki,
            "ns_talk": ns_talk,
            "ns_userTalk": ns_userTalk,
            "ns_content": ns_content,
            "weightIJ": weightij,
            "NIJ": nij
        }])

        st.subheader("📥 Input Data")
        st.dataframe(input_data)

        # ✅ Decode predicted gender
        prediction = model.predict(input_data)[0]
        decoded_gender = label_encoder.inverse_transform([prediction])[0]
        st.success(f"🧠 Predicted Gender: **{decoded_gender}**")

    except Exception as e:
        st.error(f"⚠️ Prediction failed: {e}")
