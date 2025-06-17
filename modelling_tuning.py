# modelling_tuning.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn
import os
from datetime import datetime

def load_processed_data(data_path=r"C:\Users\Ika Rachmawati\ML_PROJECT_NIDA\Membangun_model\housing_preprocessed"):
    """Load processed data with validation"""
    try:
        X_train = pd.read_csv(os.path.join(data_path, "X_train.csv"))
        X_test = pd.read_csv(os.path.join(data_path, "X_test.csv"))
        y_train = pd.read_csv(os.path.join(data_path, "y_train.csv")).iloc[:, 0]
        y_test = pd.read_csv(os.path.join(data_path, "y_test.csv")).iloc[:, 0]
        
        print("✅ Data loaded successfully")
        print(f"📊 X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        
        # Validate no string data remains
        assert all(X_train.dtypes != object), "String data detected in features!"
        
        return X_train, X_test, y_train, y_test
    except Exception as e:
        print(f"❌ Error loading data: {str(e)}")
        raise

def train_with_tuning(X_train, y_train):
    """Hyperparameter tuning with GridSearchCV"""
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    model = RandomForestRegressor(random_state=42, n_jobs=-1)
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=3,
        scoring='neg_mean_squared_error',
        verbose=2
    )
    
    print("🔍 Starting hyperparameter tuning...")
    grid_search.fit(X_train, y_train)
    print("🎯 Tuning completed!")
    
    return grid_search.best_estimator_, grid_search.best_params_

def evaluate_model(model, X_test, y_test):
    """Comprehensive model evaluation"""
    predictions = model.predict(X_test)
    metrics = {
        'rmse': np.sqrt(mean_squared_error(y_test, predictions)),
        'mae': mean_absolute_error(y_test, predictions),
        'r2': r2_score(y_test, predictions),
        'avg_price_error': np.mean(np.abs(predictions - y_test) / y_test) * 100
    }
    return metrics

def main():
    # MLflow setup
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("Housing Price Prediction - Tuned")
    
    with mlflow.start_run(run_name=f"tuning_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        # Load data
        X_train, X_test, y_train, y_test = load_processed_data()
        
        # Train with tuning
        model, best_params = train_with_tuning(X_train, y_train)
        
        # Log parameters and metrics
        mlflow.log_params(best_params)
        metrics = evaluate_model(model, X_test, y_test)
        mlflow.log_metrics(metrics)
        
        # Log model with custom signature
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="tuned_model",
            input_example=X_train.iloc[:1],
            signature=mlflow.models.infer_signature(X_train, y_train)
        )
        
        # Print results
        print("\n📈 Best Parameters:")
        for k, v in best_params.items():
            print(f"{k}: {v}")
        
        print("\n📊 Evaluation Metrics:")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")

if __name__ == "__main__":
    main()