import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

print("Loading Master Kaggle Dataset...")
data_path = os.path.join("data", "master_kaggle_dataset.csv")
df = pd.read_csv(data_path)

X = df[["RMS_Voltage", "Peak_Voltage", "THD_Percentage"]]
y = df["Fault_Type"] # The AI now learns the string names directly!

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training Advanced AI on {len(X_train)} real-world samples...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

predictions = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print(f"\n--- Real-World AI Evaluation ---")
print(f"Overall Accuracy: {accuracy * 100:.2f}%")

model_path = os.path.join("models", "rf_fault_classifier.pkl")
joblib.dump(rf_model, model_path)
print(f"New AI Brain saved to {model_path}!")