# Heart Disease Prediction MLOps Project

## Overview

This project implements an end-to-end MLOps pipeline for predicting heart disease using machine learning.

The solution covers the complete machine learning lifecycle:

- Data preprocessing
- Exploratory Data Analysis (EDA)
- Feature engineering
- Model training and hyperparameter tuning
- MLflow experiment tracking
- Champion model selection
- Model packaging and reproducibility
- Unit testing
- FastAPI model serving
- Docker containerization
- CI/CD with GitHub Actions
- Kubernetes deployment manifests
- API monitoring and logging

---

## Project Structure

```text
heart-disease-mlops/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── data/
│   ├── raw/
│   │   ├── processed.cleveland.data
│   │   ├── processed.hungarian.data
│   │   ├── processed.switzerland.data
│   │   └── processed.va.data
│   │
│   └── processed/
│       └── heart_disease_cleaned.csv
│
├── models/
│   └── champion_model.pkl
│
├── notebook/
│   └── Heart_Disease_Analysis.ipynb
│
├── plots/
│   └── model_comparison.png
│
├── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── predict.py
│   ├── schemas.py
│   └── app.py
│
├── tests/
│   ├── test_model.py
│   └── test_api.py
│
├── k8s/
│   ├── deployment.yaml
│   └── service.yaml
│
├── mlruns/
│
├── Dockerfile
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Dataset

The project uses the UCI Heart Disease dataset composed of:

- Cleveland
- Hungarian
- Switzerland
- VA Long Beach

datasets.

Raw files are stored in:

```text
data/raw/
```

Processed datasets are stored in:

```text
data/processed/
```

---

## Environment Setup

Create a virtual environment:

```bash
python -m venv venv
```

Activate environment:

### Windows

```bash
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Data Preprocessing

Generate cleaned dataset:

```bash
python src/preprocess.py
```

Output:

```text
data/processed/heart_disease_cleaned.csv
```

Processing includes:

- Dataset combination
- Missing value handling
- Type conversion
- Binary target conversion
- Data cleaning

---

## Model Training

Train and evaluate machine learning models:

```bash
python src/train.py
```

Models:

- Logistic Regression
- Random Forest

Techniques:

- GridSearchCV
- Hyperparameter tuning
- Cross Validation
- MLflow Tracking

Output:

```text
models/champion_model.pkl
```

---

## Experiment Tracking

MLflow is used for experiment tracking.

Tracked information:

- Parameters
- Metrics
- Artifacts
- Model performance
- Cross-validation results

Launch MLflow UI:

```bash
mlflow ui
```

Access:

```text
http://127.0.0.1:5000
```

---

## Local Prediction

Run prediction script:

```bash
python src/predict.py
```

Example Output:

```text
Prediction: 1
Probability: 0.9268
```

---

## API Deployment

Run FastAPI server:

```bash
uvicorn src.app:app --reload
```

API Documentation:

```text
http://127.0.0.1:8000/docs
```

Interactive Swagger UI is automatically generated.

---

## API Endpoints

### Home Endpoint

```http
GET /
```

Response:

```json
{
  "message": "Heart Disease Predictor API"
}
```

### Prediction Endpoint

```http
POST /predict
```

Example Request:

```json
{
  "age": 55,
  "sex": 1,
  "cp": 2,
  "trestbps": 140,
  "chol": 250,
  "fbs": 0,
  "restecg": 1,
  "thalach": 150,
  "exang": 0,
  "oldpeak": 1.5,
  "slope": 2,
  "ca": 0,
  "thal": 2
}
```

Example Response:

```json
{
  "prediction": 1,
  "probability": 0.9268
}
```

---

## Monitoring and Logging

API requests are logged to:

```text
api.log
```

Metrics endpoint:

```http
GET /metrics
```

Response:

```json
{
  "total_requests": 10,
  "positive_predictions": 6,
  "negative_predictions": 4
}
```

Tracked metrics include:

- Total requests
- Positive predictions
- Negative predictions

---

## Unit Testing

Execute tests using PyTest:

```bash
pytest tests/
```

Tests cover:

- Model loading
- Prediction generation
- API availability
- API response validation

---

## Docker

Build Docker image:

```bash
docker build -t heart-disease-api .
```

Run container:

```bash
docker run -p 8000:8000 heart-disease-api
```

Verify service:

```text
http://localhost:8000/docs
```

---

## CI/CD Pipeline

GitHub Actions automates:

- Dependency installation
- Unit testing
- FastAPI validation
- Docker build verification

Workflow file:

```text
.github/workflows/ci.yml
```

Pipeline automatically runs on:

- Push to main
- Pull requests

---

## Kubernetes Deployment

Deployment manifests are provided in:

```text
k8s/
```

Deploy application:

```bash
kubectl apply -f k8s/
```

Resources include:

- Deployment
- Service

Deployment YAML:

```text
k8s/deployment.yaml
```

Service YAML:

```text
k8s/service.yaml
```

---

## Model Evaluation Metrics

The models are evaluated using:

- Accuracy
- Precision
- Recall
- ROC-AUC
- Cross Validation ROC-AUC

---

## Champion Model

The best-performing model is automatically selected based on ROC-AUC score and stored as:

```text
models/champion_model.pkl
```

This model is used by:

- predict.py
- FastAPI application
- Docker container
- Kubernetes deployment

---

## How to Run Complete Pipeline

```bash
python src/preprocess.py

python src/train.py

python src/predict.py

uvicorn src.app:app --reload
```

---

## Author

Ayush Sultania

MLOps Assignment 2026