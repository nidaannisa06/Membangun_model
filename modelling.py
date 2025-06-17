import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn
import dagshub
import pickle

dagshub.init(repo_owner='nidaannisa06', repo_name='Membangun_Model', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/nidaannisa06/Membangun_Model.mlflow")
def load_processed_data(data_dir="./housing-preprocessing"):
    """Load processed data with correct relative paths"""
    try:
        X_train = pd.read_csv(os.path.join(data_dir, "X_train.csv"))
        X_test = pd.read_csv(os.path.join(data_dir, "X_test.csv"))
        y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv")).iloc[:, 0]
        y_test = pd.read_csv(os.path.join(data_dir, "y_test.csv")).iloc[:, 0]
        
        print("Data loaded successfully:")
        print(f"X_train shape: {X_train.shape}")
        print(f"y_train shape: {y_train.shape}")
        
        return X_train, X_test, y_train, y_test
    except Exception as e:
        print(f"Error loading processed data: {str(e)}")
        raise

def train_model(X_train, y_train):
    """Train Random Forest model"""
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1  # Use all available cores
    )
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    return rmse, mae, r2

def main():
    # Set tracking URI
    #mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("Housing Price Prediction")
    
    # Enable autologging
    #mlflow.sklearn.autolog()
    
    with mlflow.start_run():
        # Load processed data
        X_train, X_test, y_train, y_test = load_processed_data()
        
        # Log data parameters
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("test_samples", len(X_test))
        
        # Train model
        model = train_model(X_train, y_train)
        
        # Evaluate
        rmse, mae, r2 = evaluate_model(model, X_test, y_test)
        
        # Log metrics
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2_score", r2)
        
        # Log model
       # mlflow.sklearn.log_model(model, "random_forest_model")
        model_filename = "random_forest_model.pkl" # Nama file model yang akan disimpan
        with open(model_filename, "wb") as f:
            pickle.dump(model, f)
    print(f"Model saved locally to {model_filename}")
    
    mlflow.log_artifact(model_filename, "model_artefacts") # "model_artefacts" adalah sub-path di MLflow UI
    print(f"Model logged as artifact to MLflow/DagsHub at '{model_filename}'")

    print(f"Training complete. RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.2f}")

if __name__ == "__main__":
    main()
    with open ("dagshub.txt", "w") as f:
        f.write("https://dagshub.com/nidaannisa06/Membangun_Model.mlflow")
    print("dagshub run succesfully")