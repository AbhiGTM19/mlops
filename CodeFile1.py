import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn
from sklearn.metrics import precision_score, recall_score, f1_score, ConfusionMatrixDisplay

df = pd.read_csv("Iris.csv")
df.sample(4)

X = df[["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"]]
y = df["Species"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

# Initialize Random Forest Classifier
rf = RandomForestClassifier(random_state=42)

# Set up GridSearchCV with 5-fold cross-validation
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

# Fit GridSearchCV
grid_search.fit(X_train, y_train)

# Report results
print("Best Hyperparameters:", grid_search.best_params_)
print("Best Cross-Validated Accuracy:", grid_search.best_score_)

# Evaluate on test set
best_model = grid_search.best_estimator_
test_accuracy = best_model.score(X_test, y_test)
print("Test Set Accuracy:", test_accuracy)

with mlflow.start_run(run_name="HyperParameterTuning"):
    grid_search.fit(X_train, y_train)

    mlflow.log_params(grid_search.best_params_) #log parametes to ML flow
    mlflow.log_metric("best_cv_accuracy", grid_search.best_score_)

    # Evaluate on test set and log test accuracy
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    test_accuracy = best_model.score(X_test, y_test)
    mlflow.log_metric("test_accuracy", test_accuracy)

    # Log the best model
    mlflow.sklearn.log_model(best_model, "best_rf_model")

    # Print results
    print("Best Hyperparameters:", grid_search.best_params_)
    print("Best Cross-Validated Accuracy:", grid_search.best_score_)
    print("Test Set Accuracy:", test_accuracy)

 # Log additional test set metrics
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    mlflow.log_metric("test_precision_weighted", precision)
    mlflow.log_metric("test_recall_weighted", recall)
    mlflow.log_metric("test_f1_weighted", f1)