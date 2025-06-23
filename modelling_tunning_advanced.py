# modelling_tuning.py

import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
import mlflow
import mlflow.sklearn
import pickle
import dagshub
import logging
from datetime import datetime

# Configure logging for better visibility in the console output
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- DagsHub Initialization for Online MLflow Tracking (Advanced Criteria) ---
# Initialize DagsHub. This sets up the MLFLOW_TRACKING_URI and
# necessary credentials for DagsHub. This needs to be done once per session.
try:
    # IMPORTANT: Ensure 'repo_owner' and 'repo_name' match your DagsHub project
    dagshub.init(repo_owner='nidaannisa06', repo_name='membangun_model_advanced', mlflow=True)
    logger.info("DagsHub MLflow initialized successfully for online tracking.")
except Exception as e:
    logger.error(f"Failed to initialize DagsHub MLflow. Make sure your DagsHub token is configured and you have access to the repository. Error: {e}")
    # For Advanced criteria, DagsHub setup MUST be successful. Raise an error to stop.
    raise

# Set the experiment name for DagsHub runs
mlflow.set_experiment("Housing Price Prediction - Tuned Advanced DagsHub")

def load_processed_data(data_path="housing_preprocessing"):
    """
    Loads processed data (X_train, X_test, y_train, y_test) from CSV files.
    This function expects the 'housing_preprocessing' directory to contain
    'X_train.csv', 'X_test.csv', 'y_train.csv', and 'y_test.csv'.
    """
    try:
             
        X_train = pd.read_csv(os.path.join(data_path, "X_train.csv"))
        X_test = pd.read_csv(os.path.join(data_path, "X_test.csv"))
        
        # Ensure 'price' column is read correctly for y_train and y_test
        y_train = pd.read_csv(os.path.join(data_path, "y_train.csv"))['price']
        y_test = pd.read_csv(os.path.join(data_path, "y_test.csv"))['price']

        logger.info("✅ Data loaded successfully.")
        logger.info(f"📊 X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        
        # Assertions to ensure data consistency between features and targets
        assert X_train.shape[0] == y_train.shape[0], f"X_train ({X_train.shape[0]}) and y_train ({y_train.shape[0]}) have inconsistent samples!"
        assert X_test.shape[0] == y_test.shape[0], f"X_test ({X_test.shape[0]}) and y_test ({y_test.shape[0]}) have inconsistent samples!"
        
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logger.error(f"❌ Error loading processed data: {str(e)}")
        # Print full traceback for detailed debugging information
        import traceback
        traceback.print_exc()
        raise # Re-raise the exception to stop execution if data loading fails critically

def train_with_tuning(X_train, y_train):
    """
    Performs hyperparameter tuning using GridSearchCV on RandomForestRegressor.
    Returns the best estimator and its best parameters.
    """
    # Define the parameter grid for GridSearchCV
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    # Initialize the base model
    base_model = RandomForestRegressor(random_state=42, n_jobs=-1)
    
    # Setup GridSearchCV
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=3, # 3-fold cross-validation
        scoring='neg_mean_squared_error', # Optimize for lowest MSE
        verbose=2, # Show progress
        n_jobs=-1 # Use all available cores for Grid Search
    )
    
    logger.info("🔍 Starting hyperparameter tuning with GridSearchCV...")
    grid_search.fit(X_train, y_train)
    logger.info("🎯 Hyperparameter tuning completed!")
    
    logger.info(f"Best parameters found: {grid_search.best_params_}")
    logger.info(f"Best cross-validation score (negative MSE): {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_, grid_search.best_params_

def evaluate_model(model, X_test, y_test):
    """
    Evaluates model performance and calculates standard regression metrics.
    Returns calculated metrics and the predictions for additional metric calculation.
    """
    predictions = model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    logger.info(f"Model evaluation complete. RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.2f}")
    return rmse, mae, r2, predictions # Return predictions to use for additional metrics

def calculate_additional_metrics(y_true, y_pred):
    """
    Calculates additional regression metrics as required by the Advanced criteria
    (at least 2 beyond RMSE, MAE, R2).
    """
    metrics = {}

    # 1. Mean Absolute Percentage Error (MAPE)
    # Important: Filter out zero true values to avoid division by zero errors
    y_true_non_zero = y_true[y_true != 0]
    y_pred_non_zero = y_pred[y_true != 0]
    
    if len(y_true_non_zero) > 0:
        mape = np.mean(np.abs((y_true_non_zero - y_pred_non_zero) / y_true_non_zero)) * 100
        metrics['mape'] = mape
        logger.info(f"Calculated MAPE: {mape:.2f}%")
    else:
        metrics['mape'] = float('nan') # Log NaN if MAPE cannot be calculated
        logger.warning("Cannot calculate MAPE as all true values are zero in the test set.")

    # 2. Explained Variance Score
    # This metric measures the proportion to which a model accounts for the variation of a given data set.
    explained_var = explained_variance_score(y_true, y_pred)
    metrics['explained_variance_score'] = explained_var
    logger.info(f"Calculated Explained Variance Score: {explained_var:.2f}")

    # Optional: You could add more metrics here if desired, e.g., Max Error
    # metrics['max_error'] = max_error(y_true, y_pred)
    # logger.info(f"Calculated Max Error: {metrics['max_error']:.2f}")

    return metrics

def save_model_locally(model, model_path="model_artifacts"):
    """
    Save model locally as a backup and for potential manual artifact logging
    """
    os.makedirs(model_path, exist_ok=True)
    model_file_path = os.path.join(model_path, "tuned_random_forest_model.pkl")
    
    with open(model_file_path, 'wb') as f:
        pickle.dump(model, f)
    
    logger.info(f"Model saved locally at: {model_file_path}")
    return model_file_path

def main():
    # Advanced Criteria
    # Disable autologging as manual logging is explicitly required for this level.
    # mlflow.sklearn.autolog() # <-- This line MUST remain commented out/removed
    
    # Start an MLflow run. 'as run' captures the run object for logging its ID/URI.
    # Using a unique run name for better organization in DagsHub UI
    run_name = f"tuning_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    with mlflow.start_run(run_name=run_name) as run:
        logger.info(f"MLflow Run started with ID: {run.info.run_id}")
        logger.info(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")

        # Load the preprocessed data
        X_train, X_test, y_train, y_test = load_processed_data()
        
        # Perform Hyperparameter Tuning and get best model
        best_model, best_params = train_with_tuning(X_train, y_train)
        
        # Manual Logging of Parameters (Advanced Criteria)
        # Log the best hyperparameters found by GridSearchCV
        mlflow.log_params(best_params)
        logger.info("Manually logged best hyperparameters from tuning.")

        # Log data-related parameters for better traceability
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("test_samples", len(X_test))
        mlflow.log_param("features_count", X_train.shape[1])
        logger.info("Manually logged data parameters.")

        # Evaluate the best model and get predictions
        rmse, mae, r2, predictions = evaluate_model(best_model, X_test, y_test)
        
        # Manual Logging of Core Metrics (Advanced Criteria)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2_score", r2)
        logger.info("Manually logged core evaluation metrics (RMSE, MAE, R2).")

        # Calculate and Log Additional Metrics (Advanced Criteria)
        additional_metrics = calculate_additional_metrics(y_test, predictions)
        for metric_name, metric_value in additional_metrics.items():
            mlflow.log_metric(metric_name, metric_value)
        logger.info("Manually logged additional evaluation metrics (MAPE, Explained Variance).")

        # Save model locally first (as a backup)
        model_file_path = save_model_locally(best_model)
        
        # Manual Logging of the Model to DagsHub
        try:
            # Log the best model using mlflow.sklearn.log_model
            # Keep it simple to avoid potential DagsHub compatibility issues if any
            mlflow.sklearn.log_model(
                sk_model=best_model, 
                artifact_path="tuned_random_forest_model",
                # It's good practice to provide an input example and signature
                # if you plan to deploy with MLflow's pyfunc, but can be removed
                # if it causes issues with DagsHub directly logging the model artifact.
                # input_example=X_train.iloc[:1], 
                # signature=mlflow.models.infer_signature(X_train, y_train)
            )
            logger.info("Successfully logged the best RandomForestRegressor model as an artifact to DagsHub.")
        except Exception as e:
            logger.error(f"Failed to log model to DagsHub via mlflow.sklearn.log_model: {e}")
            logger.warning("Attempting to log the locally saved model file as a generic artifact instead.")
            
            # Fallback: Log the locally saved .pkl file as a generic artifact
            try:
                mlflow.log_artifact(model_file_path, "model_backup_pkl")
                logger.info("Successfully logged local .pkl model file as artifact backup.")
            except Exception as backup_error:
                logger.error(f"Failed to log backup model artifact: {backup_error}")
        
        # --- Log additional non-model artifacts (e.g., info file) ---
        try:
            # Create a dictionary of key model and run information
            run_info_summary = {
                "Run ID": run.info.run_id,
                "Experiment Name": mlflow.active_run().info.experiment_name,
                "Model Type": "RandomForestRegressor",
                "Best Parameters": best_params,
                "Core Metrics": {"RMSE": rmse, "MAE": mae, "R2": r2},
                "Additional Metrics": additional_metrics,
                "Training Samples": len(X_train),
                "Test Samples": len(X_test),
                "Features Count": X_train.shape[1],
                "Timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Save this information to a file
            info_file_dir = "run_artifacts"
            os.makedirs(info_file_dir, exist_ok=True)
            info_file_path = os.path.join(info_file_dir, f"run_summary_{run.info.run_id}.txt")
            with open(info_file_path, 'w') as f:
                for key, value in run_info_summary.items():
                    f.write(f"{key}: {value}\n")
            
            # Log the info file as an artifact
            mlflow.log_artifact(info_file_path, "run_metadata")
            logger.info("Successfully logged run summary information as artifact.")
            
        except Exception as e:
            logger.warning(f"Failed to log run summary artifact: {e}")
        
        # --- Final summary output to console ---
        print(f"\n--- Training and Logging Complete ---")
        print(f"MLflow Run ID: {run.info.run_id}")
        print(f"Best Parameters: {best_params}")
        print(f"Core Metrics: RMSE: {rmse:.2f}, MAE: {mae:.2f}, R2: {r2:.2f}")
        for name, value in additional_metrics.items():
            print(f"Additional Metric - {name.upper()}: {value:.2f}")
        
        # --- Fixed DagsHub URL construction ---
        # Simplified URL construction without using dagshub.auth.get_username()
        print(f"\n✨ MLflow Run logged to DagsHub. View details:")
        print(f"   DagsHub Project MLflow UI: https://dagshub.com/nidaannisa06/membangun_model_advanced/mlflow")
        print(f"   Run ID: {run.info.run_id}")
        print(f"   Experiment ID: {mlflow.active_run().info.experiment_id}")
        print(f"Model saved locally at: {model_file_path}")

        # --- Write DagsHub URL to file ---
        try:
            with open("dagshub_mlflow_url.txt", "w") as f:
                f.write(f"https://dagshub.com/nidaannisa06/membangun_model_advanced/mlflow\n")
                f.write(f"Run ID: {run.info.run_id}\n")
                f.write(f"Experiment ID: {mlflow.active_run().info.experiment_id}\n")
                f.write(f"After running this script, check your DagsHub MLflow UI for the latest run: https://dagshub.com/nidaannisa06/membangun_model_advanced/mlflow")
            logger.info("Information about DagsHub MLflow URL written to dagshub_mlflow_url.txt")
        except Exception as e:
            logger.warning(f"Failed to write DagsHub URL file: {e}")

if __name__ == "__main__":
    main()