# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib

# Load data
df = pd.read_csv('data.csv')
df = df.drop(columns=['firstDay', 'lastDay'])

# Encode C_api
label_enc = LabelEncoder()
df['C_api'] = label_enc.fit_transform(df['C_api'])

# Train/test split
X = df.drop(columns=['gender'])
y = df['gender']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model & encoder
joblib.dump(model, 'model.pkl')
joblib.dump(label_enc, 'c_api_encoder.pkl')
print('Model saved. Accuracy:', accuracy_score(y_test, model.predict(X_test)))
