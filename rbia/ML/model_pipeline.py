import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_validate
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import make_scorer, accuracy_score, f1_score, precision_score, recall_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings("ignore")

# ==========================================================
# CONFIGURATION — CHANGE ONLY THESE
# ==========================================================
CSV_PATH = "your_data.csv"          # Path to your dataset
TARGET_COLUMN = "target"            # Name of the target column
KFOLD_SPLITS = 5                    # Number of folds for cross-validation
# ==========================================================

# Load data
print("📂 Loading dataset...")
data = pd.read_csv(CSV_PATH)

if TARGET_COLUMN not in data.columns:
    raise ValueError(f"Target column '{TARGET_COLUMN}' not found in CSV file.")

X = data.drop(columns=[TARGET_COLUMN])
y = data[TARGET_COLUMN]

# Separate categorical and numerical columns
categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

# Preprocessing for numeric features: impute missing and scale
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

# Preprocessing for categorical features: impute and one-hot encode
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# Combine preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols)
    ]
)

# Define models to evaluate
models = {
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "XGBoost": XGBClassifier(
        n_estimators=200,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42
    ),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}

# Scoring metrics
scoring = {
    "accuracy": make_scorer(accuracy_score),
    "f1": make_scorer(f1_score, average="weighted"),
    "precision": make_scorer(precision_score, average="weighted"),
    "recall": make_scorer(recall_score, average="weighted"),
}

# K-Fold setup
kf = KFold(n_splits=KFOLD_SPLITS, shuffle=True, random_state=42)

results = []

print("⚙️ Running model evaluations...\n")

for name, model in models.items():
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])
    
    cv_results = cross_validate(
        pipeline,
        X, y,
        cv=kf,
        scoring=scoring,
        n_jobs=-1
    )

    result = {
        "Model": name,
        "Accuracy": np.mean(cv_results["test_accuracy"]),
        "F1-Score": np.mean(cv_results["test_f1"]),
        "Precision": np.mean(cv_results["test_precision"]),
        "Recall": np.mean(cv_results["test_recall"])
    }
    results.append(result)

# Create sorted results table
results_df = pd.DataFrame(results).sort_values(by="F1-Score", ascending=False)

print("🏁 Cross-validation results:\n")
print(results_df.to_string(index=False))

# Optional: save results to CSV
results_df.to_csv("model_scores.csv", index=False)
print("\n📄 Results saved to model_scores.csv")
