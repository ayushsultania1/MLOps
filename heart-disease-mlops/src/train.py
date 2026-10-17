import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    cross_val_score
)

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score
)

# -----------------------------
# Load Data
# -----------------------------
df = pd.read_csv(
    "data/processed/heart_disease_cleaned.csv"
)

X = df.drop("target", axis=1)
y = df["target"]

# -----------------------------
# Train Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------------
# Scaling for Logistic Regression
# -----------------------------
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# MLflow Experiment
# -----------------------------
mlflow.set_experiment(
    "Heart_Disease_Prediction"
)

# =====================================================
# Logistic Regression
# =====================================================

lr_grid = GridSearchCV(
    LogisticRegression(
        max_iter=1000,
        random_state=42
    ),
    {
        "C": [0.01, 0.1, 1, 10, 100]
    },
    cv=5,
    scoring="roc_auc"
)

lr_grid.fit(
    X_train_scaled,
    y_train
)

best_lr = lr_grid.best_estimator_

lr_pred = best_lr.predict(
    X_test_scaled
)

lr_prob = best_lr.predict_proba(
    X_test_scaled
)[:, 1]

lr_accuracy = accuracy_score(
    y_test,
    lr_pred
)

lr_precision = precision_score(
    y_test,
    lr_pred
)

lr_recall = recall_score(
    y_test,
    lr_pred
)

lr_auc = roc_auc_score(
    y_test,
    lr_prob
)

lr_cv = cross_val_score(
    best_lr,
    scaler.fit_transform(X),
    y,
    cv=5,
    scoring="roc_auc"
).mean()

with mlflow.start_run(
    run_name="Logistic_Regression"
):

    mlflow.log_param(
        "model",
        "LogisticRegression"
    )

    mlflow.log_param(
        "C",
        lr_grid.best_params_["C"]
    )

    mlflow.log_metric(
        "accuracy",
        lr_accuracy
    )

    mlflow.log_metric(
        "precision",
        lr_precision
    )

    mlflow.log_metric(
        "recall",
        lr_recall
    )

    mlflow.log_metric(
        "roc_auc",
        lr_auc
    )

    mlflow.log_metric(
        "cv_roc_auc",
        lr_cv
    )

print("Logistic Regression completed")

# =====================================================
# Random Forest
# =====================================================

rf_grid = GridSearchCV(
    RandomForestClassifier(
        random_state=42
    ),
    {
        "n_estimators": [100, 200, 300],
        "max_depth": [5, 10, 15, None]
    },
    cv=5,
    scoring="roc_auc"
)

rf_grid.fit(
    X_train,
    y_train
)

best_rf = rf_grid.best_estimator_

rf_pred = best_rf.predict(
    X_test
)

rf_prob = best_rf.predict_proba(
    X_test
)[:, 1]

rf_accuracy = accuracy_score(
    y_test,
    rf_pred
)

rf_precision = precision_score(
    y_test,
    rf_pred
)

rf_recall = recall_score(
    y_test,
    rf_pred
)

rf_auc = roc_auc_score(
    y_test,
    rf_prob
)

rf_cv = cross_val_score(
    best_rf,
    X,
    y,
    cv=5,
    scoring="roc_auc"
).mean()

with mlflow.start_run(
    run_name="Random_Forest"
):

    mlflow.log_param(
        "model",
        "RandomForest"
    )

    mlflow.log_param(
        "n_estimators",
        rf_grid.best_params_["n_estimators"]
    )

    mlflow.log_param(
        "max_depth",
        rf_grid.best_params_["max_depth"]
    )

    mlflow.log_metric(
        "accuracy",
        rf_accuracy
    )

    mlflow.log_metric(
        "precision",
        rf_precision
    )

    mlflow.log_metric(
        "recall",
        rf_recall
    )

    mlflow.log_metric(
        "roc_auc",
        rf_auc
    )

    mlflow.log_metric(
        "cv_roc_auc",
        rf_cv
    )

print("Random Forest completed")

# =====================================================
# Champion Selection
# =====================================================

if rf_auc > lr_auc:
    champion_model = best_rf
    champion_name = "Random Forest"
    champion_score = rf_auc
else:
    champion_model = best_lr
    champion_name = "Logistic Regression"
    champion_score = lr_auc

joblib.dump(
    champion_model,
    "models/champion_model.pkl"
)

print("\n===================")
print("Champion Model")
print("===================")
print("Model :", champion_name)
print("ROC-AUC :", round(champion_score, 4))
print("Saved -> models/champion_model.pkl")