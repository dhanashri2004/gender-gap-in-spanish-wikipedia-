import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# === 1. Load dataset ===
df = pd.read_csv("data.csv")

# === 2. Encode categorical features ===
c_api_encoder = LabelEncoder()
df["C_api"] = c_api_encoder.fit_transform(df["C_api"])
joblib.dump(c_api_encoder, "c_api_encoder.pkl")

# === 3. Encode the label ===
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df["gender"])
joblib.dump(label_encoder, "label_encoder.pkl")  # ✅ Save label encoder

# === 4. Prepare feature columns ===
X = df.drop(columns=["gender"])  # Drop target
X = X[["C_api", "C_man", "E_NEds", "E_Bpag", "NEds", "NDays", "NActDays",
       "NPages", "NPcreated", "pagesWomen", "wikiprojWomen",
       "ns_user", "ns_wikipedia", "ns_talk", "ns_userTalk",
       "ns_content", "weightIJ", "NIJ"]]  # Columns used for training

# === 5. Train/test split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === 6. Train model ===
model = RandomForestClassifier()
model.fit(X_train, y_train)

# === 7. Save the model ===
joblib.dump(model, "model.pkl")

print("✅ Model and encoders saved.")
