import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn
import pickle
from mlflow.pyfunc import PythonModel
# import dagshub # No need to import dagshub if only tracking locally

# Remove the dagshub.init() line as we'll only be tracking locally
# dagshub.init(repo_owner='nidaannisa06', repo_name='testingNidaannisa19', mlflow=True)

# This is the requested configuration: track to your local MLflow UI
mlflow.set_tracking_uri("http://127.0.0.1:5000") # Logs to your local MLflow UI
mlflow.set_experiment("Housing Price Prediction - Basic Local") # Experiment name for local runs

def load_processed_data(data_path="housing_preprocessing"):
    """
    Load processed data (X_train, X_test, y_train, y_test) from CSV files.
    """
    try:
        X_train = pd.read_csv(os.path.join(data_path, "X_train.csv"))
        X_test = pd.read_csv(os.path.join(data_path, "X_test.csv"))
        
        y_train = pd.read_csv(os.path.join(data_path, "y_train.csv"))['price']
        y_test = pd.read_csv(os.path.join(data_path, "y_test.csv"))['price']

        print("✅ Data loaded successfully")
        print(f"📊 X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        
        assert X_train.shape[0] == y_train.shape[0], f"X_train ({X_train.shape[0]}) and y_train ({y_train.shape[0]}) have inconsistent samples!"
        assert X_test.shape[0] == y_test.shape[0], f"X_test ({X_test.shape[0]}) and y_test ({y_test.shape[0]}) have inconsistent samples!"
        
        return X_train, X_test, y_train, y_test
    except Exception as e:
        print(f"❌ Error loading processed data: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

def train_model(X_train, y_train):
    """Train Random Forest model"""
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
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

# --- Custom PythonModel class for MLflow ---
class HousingPriceModel(PythonModel):
    def load_context(self, context):
        print("DEBUG: Custom PyFunc: Loading model from context.")
        self.model = pickle.load(open(context.artifacts["model_pkl_file"], "rb"))
        print("DEBUG: Custom PyFunc: Model loaded successfully into context.")

    def predict(self, context, model_input):
        print(f"DEBUG: Custom PyFunc: Received prediction request. Input shape: {model_input.shape}")
        if not isinstance(model_input, pd.DataFrame):
            raise TypeError("Input to predict must be a pandas DataFrame.")
        
        predictions = self.model.predict(model_input)
        print(f"DEBUG: Custom PyFunc: Prediction successful. Output: {predictions.tolist()[:5]}...")
        return predictions

def main():
    # Enable autologging for scikit-learn models (REQUIRED for Basic criteria)
    mlflow.sklearn.autolog() 
    
    with mlflow.start_run():
        X_train, X_test, y_train, y_test = load_processed_data()
        
        # Parameters will be logged automatically by autolog. Manual log_param calls are REMOVED.
        
        model = train_model(X_train, y_train) # Training the model - autolog will capture params and metrics

        # Evaluation is still performed, but metrics will be logged automatically by autolog.
        # Manual log_metric calls are REMOVED.
        rmse, mae, r2 = evaluate_model(model, X_test, y_test) 
        
        # Model will be logged automatically by autolog.
        
        print(f"Training complete. RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.2f}")
        # This URL will point to local as tracking_uri is set to localhost
        print(f"MLflow Run logged to local UI. View at: {mlflow.active_run().info.artifact_uri}")

if __name__ == "__main__":
    main()