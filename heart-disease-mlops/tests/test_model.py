import joblib
import pandas as pd


def test_model_prediction():

    model = joblib.load(
        "models/champion_model.pkl"
    )

    sample = pd.DataFrame({
        "age":[55],
        "sex":[1],
        "cp":[2],
        "trestbps":[140],
        "chol":[250],
        "fbs":[0],
        "restecg":[1],
        "thalach":[150],
        "exang":[0],
        "oldpeak":[1.5],
        "slope":[2],
        "ca":[0],
        "thal":[2]
    })

    prediction = model.predict(sample)

    assert prediction[0] in [0, 1]