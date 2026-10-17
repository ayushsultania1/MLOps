import joblib
import pandas as pd

model = joblib.load(
    "models/champion_model.pkl"
)

sample_patient = pd.DataFrame({
    "age": [55],
    "sex": [1],
    "cp": [2],
    "trestbps": [140],
    "chol": [250],
    "fbs": [0],
    "restecg": [1],
    "thalach": [150],
    "exang": [0],
    "oldpeak": [1.5],
    "slope": [2],
    "ca": [0],
    "thal": [2]
})

prediction = model.predict(sample_patient)[0]

probability = model.predict_proba(
    sample_patient
)[0][1]

print(f"Prediction: {prediction}")
print(f"Probability: {probability:.4f}")