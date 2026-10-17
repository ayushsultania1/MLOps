import pandas as pd
import numpy as np

columns = [
    'age','sex','cp','trestbps','chol',
    'fbs','restecg','thalach',
    'exang','oldpeak','slope',
    'ca','thal','target'
]

files = [
    "data/raw/processed.cleveland.data",
    "data/raw/processed.hungarian.data",
    "data/raw/processed.switzerland.data",
    "data/raw/processed.va.data"
]

dfs = []

for file in files:
    df = pd.read_csv(file, header=None, names=columns)
    dfs.append(df)

heart_df = pd.concat(dfs, ignore_index=True)

heart_df.replace('?', np.nan, inplace=True)

for col in heart_df.columns:
    heart_df[col] = pd.to_numeric(
        heart_df[col],
        errors='coerce'
    )

heart_df['target'] = (
    heart_df['target'] > 0).astype(int)

heart_df.fillna(
    heart_df.median(numeric_only=True),
    inplace=True
)

heart_df.to_csv(
    "data/processed/heart_disease_cleaned.csv",
    index=False
)

print("Clean dataset saved")
print(heart_df.shape)