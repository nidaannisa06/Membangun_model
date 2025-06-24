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
import matplotlib.pyplot as plt
import seaborn as sns
import json

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
mlflow.set_experiment("Housing Price Prediction - Complete Artifacts")

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
    Returns the best estimator, its best parameters, and training history.
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
    
    # Extract training history from GridSearchCV results
    training_history = {
        'param_combinations': len(grid_search.cv_results_['params']),
        'best_score': grid_search.best_score_,
        'cv_results': grid_search.cv_results_
    }
    
    return grid_search.best_estimator_, grid_search.best_params_, training_history

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

    return metrics

def create_regression_visualizations(y_true, y_pred, model, X_test, artifacts_dir="visualization_artifacts"):
    """
    Creates comprehensive visualizations for regression model evaluation.
    Returns paths to saved visualization files.
    """
    os.makedirs(artifacts_dir, exist_ok=True)
    visualization_paths = []
    
    # Set style for better-looking plots
    plt.style.use('seaborn-v0_8')
    
    # 1. Actual vs Predicted Plot (Regression equivalent of confusion matrix)
    plt.figure(figsize=(10, 8))
    plt.scatter(y_true, y_pred, alpha=0.6, color='blue', edgecolors='black', linewidth=0.5)
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    plt.xlabel('Actual Values', fontsize=12)
    plt.ylabel('Predicted Values', fontsize=12)
    plt.title('Actual vs Predicted Values\n(Regression Performance Analysis)', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add R² score as text
    r2 = r2_score(y_true, y_pred)
    plt.text(0.05, 0.95, f'R² Score: {r2:.3f}', transform=plt.gca().transAxes, 
             fontsize=12, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    actual_vs_pred_path = os.path.join(artifacts_dir, "actual_vs_predicted_plot.png")
    plt.tight_layout()
    plt.savefig(actual_vs_pred_path, dpi=300, bbox_inches='tight')
    plt.close()
    visualization_paths.append(actual_vs_pred_path)
    logger.info("✅ Created Actual vs Predicted plot")
    
    # 2. Residuals Plot
    plt.figure(figsize=(12, 6))
    residuals = y_true - y_pred
    
    # Residuals vs Predicted
    plt.subplot(1, 2, 1)
    plt.scatter(y_pred, residuals, alpha=0.6, color='green', edgecolors='black', linewidth=0.5)
    plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
    plt.xlabel('Predicted Values', fontsize=12)
    plt.ylabel('Residuals', fontsize=12)
    plt.title('Residuals vs Predicted Values', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Residuals Histogram
    plt.subplot(1, 2, 2)
    plt.hist(residuals, bins=30, alpha=0.7, color='orange', edgecolor='black')
    plt.xlabel('Residuals', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Distribution of Residuals', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    residuals_path = os.path.join(artifacts_dir, "residuals_analysis_plot.png")
    plt.tight_layout()
    plt.savefig(residuals_path, dpi=300, bbox_inches='tight')
    plt.close()
    visualization_paths.append(residuals_path)
    logger.info("✅ Created Residuals analysis plot")
    
    # 3. Feature Importance Plot
    if hasattr(model, 'feature_importances_'):
        plt.figure(figsize=(12, 8))
        feature_names = X_test.columns if hasattr(X_test, 'columns') else [f'Feature_{i}' for i in range(X_test.shape[1])]
        importances = model.feature_importances_
        
        # Sort features by importance
        indices = np.argsort(importances)[::-1]
        
        # Plot top 15 features
        top_n = min(15, len(feature_names))
        plt.bar(range(top_n), importances[indices[:top_n]], color='skyblue', edgecolor='black')
        plt.xlabel('Features', fontsize=12)
        plt.ylabel('Importance Score', fontsize=12)
        plt.title('Top Feature Importances\n(Random Forest Model)', fontsize=14, fontweight='bold')
        plt.xticks(range(top_n), [feature_names[i] for i in indices[:top_n]], rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        
        feature_importance_path = os.path.join(artifacts_dir, "feature_importance_plot.png")
        plt.tight_layout()
        plt.savefig(feature_importance_path, dpi=300, bbox_inches='tight')
        plt.close()
        visualization_paths.append(feature_importance_path)
        logger.info("✅ Created Feature Importance plot")
        
        # Save feature importance as CSV
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        feature_importance_csv_path = os.path.join(artifacts_dir, "feature_importance_analysis.csv")
        feature_importance_df.to_csv(feature_importance_csv_path, index=False)
        visualization_paths.append(feature_importance_csv_path)
        logger.info("✅ Created Feature Importance CSV")
    
    # 4. Model Performance Summary Plot
    plt.figure(figsize=(10, 6))
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    
    metrics_names = ['RMSE', 'MAE', 'R² Score']
    metrics_values = [rmse, mae, r2]
    colors = ['red', 'orange', 'green']
    
    bars = plt.bar(metrics_names, metrics_values, color=colors, alpha=0.7, edgecolor='black')
    plt.ylabel('Metric Value', fontsize=12)
    plt.title('Model Performance Metrics Summary', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, value in zip(bars, metrics_values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                 f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    performance_summary_path = os.path.join(artifacts_dir, "performance_metrics_summary.png")
    plt.tight_layout()
    plt.savefig(performance_summary_path, dpi=300, bbox_inches='tight')
    plt.close()
    visualization_paths.append(performance_summary_path)
    logger.info("✅ Created Performance Metrics Summary plot")
    
    return visualization_paths

def create_training_history_visualization(training_history, artifacts_dir="visualization_artifacts"):
    """
    Creates visualization of training/tuning history from GridSearchCV results.
    """
    os.makedirs(artifacts_dir, exist_ok=True)
    
    # Extract CV results
    cv_results = training_history['cv_results']
    
    # 1. Parameter Performance Visualization
    plt.figure(figsize=(15, 10))
    
    # Plot 1: n_estimators vs score
    plt.subplot(2, 2, 1)
    n_estimators_scores = {}
    for i, params in enumerate(cv_results['params']):
        n_est = params['n_estimators']
        score = -cv_results['mean_test_score'][i]  # Convert back to positive
        if n_est not in n_estimators_scores:
            n_estimators_scores[n_est] = []
        n_estimators_scores[n_est].append(score)
    
    n_est_values = sorted(n_estimators_scores.keys())
    avg_scores = [np.mean(n_estimators_scores[n]) for n in n_est_values]
    plt.plot(n_est_values, avg_scores, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Number of Estimators')
    plt.ylabel('Average MSE')
    plt.title('n_estimators vs Performance')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: max_depth vs score
    plt.subplot(2, 2, 2)
    max_depth_scores = {}
    for i, params in enumerate(cv_results['params']):
        depth = params['max_depth']
        depth_str = 'None' if depth is None else str(depth)
        score = -cv_results['mean_test_score'][i]
        if depth_str not in max_depth_scores:
            max_depth_scores[depth_str] = []
        max_depth_scores[depth_str].append(score)
    
    depth_labels = list(max_depth_scores.keys())
    depth_scores = [np.mean(max_depth_scores[d]) for d in depth_labels]
    plt.bar(depth_labels, depth_scores, color='orange', alpha=0.7, edgecolor='black')
    plt.xlabel('Max Depth')
    plt.ylabel('Average MSE')
    plt.title('max_depth vs Performance')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Cross-validation scores distribution
    plt.subplot(2, 2, 3)
    mean_scores = -cv_results['mean_test_score']  # Convert to positive
    std_scores = cv_results['std_test_score']
    
    plt.errorbar(range(len(mean_scores)), mean_scores, yerr=std_scores, 
                 fmt='o', capsize=5, capthick=2, alpha=0.7)
    plt.xlabel('Parameter Combination Index')
    plt.ylabel('CV Score (MSE)')
    plt.title('Cross-Validation Scores with Std')
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Best vs Worst parameter combinations
    plt.subplot(2, 2, 4)
    sorted_indices = np.argsort(mean_scores)
    best_5_scores = mean_scores[sorted_indices[:5]]
    worst_5_scores = mean_scores[sorted_indices[-5:]]
    
    x_pos = np.arange(5)
    width = 0.35
    plt.bar(x_pos - width/2, best_5_scores, width, label='Best 5', color='green', alpha=0.7)
    plt.bar(x_pos + width/2, worst_5_scores, width, label='Worst 5', color='red', alpha=0.7)
    plt.xlabel('Parameter Combination Rank')
    plt.ylabel('MSE Score')
    plt.title('Best vs Worst Parameter Combinations')
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    
    training_history_path = os.path.join(artifacts_dir, "training_history_visualization.png")
    plt.tight_layout()
    plt.savefig(training_history_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("✅ Created Training History visualization")
    return training_history_path

def create_comprehensive_model_report(model, best_params, rmse, mae, r2, additional_metrics, 
                                     training_history, X_train, X_test, artifacts_dir="visualization_artifacts"):
    """
    Creates a comprehensive model report in JSON format.
    """
    os.makedirs(artifacts_dir, exist_ok=True)
    
    report = {
        "model_info": {
            "model_type": "RandomForestRegressor",
            "training_timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "sklearn_version": "latest"
        },
        "data_info": {
            "training_samples": len(X_train),
            "test_samples": len(X_test),
            "feature_count": X_train.shape[1],
            "feature_names": list(X_train.columns) if hasattr(X_train, 'columns') else None
        },
        "hyperparameters": best_params,
        "performance_metrics": {
            "core_metrics": {
                "rmse": float(rmse),
                "mae": float(mae),
                "r2_score": float(r2)
            },
            "additional_metrics": {k: float(v) if not np.isnan(v) else None for k, v in additional_metrics.items()}
        },
        "training_details": {
            "total_parameter_combinations_tested": training_history['param_combinations'],
            "best_cv_score": float(training_history['best_score']),
            "cross_validation_folds": 3
        },
        "model_interpretation": {
            "feature_importance_available": hasattr(model, 'feature_importances_'),
            "top_5_features": None
        }
    }
    
    # Add feature importance if available
    if hasattr(model, 'feature_importances_') and hasattr(X_train, 'columns'):
        feature_names = X_train.columns
        importances = model.feature_importances_
        top_features_idx = np.argsort(importances)[::-1][:5]
        report["model_interpretation"]["top_5_features"] = [
            {"feature": feature_names[idx], "importance": float(importances[idx])} 
            for idx in top_features_idx
        ]
    
    report_path = os.path.join(artifacts_dir, "comprehensive_model_report.json")
    with open(report_path, 'w', encoding='utf-8') as f: # Added encoding='utf-8'
        json.dump(report, f, indent=2)
    
    logger.info("✅ Created Comprehensive Model Report")
    return report_path

def save_predictions_and_analysis(y_test, predictions, artifacts_dir="visualization_artifacts"):
    """
    Saves detailed prediction results and analysis.
    """
    os.makedirs(artifacts_dir, exist_ok=True)
    
    # Create predictions DataFrame
    results_df = pd.DataFrame({
        'actual_price': y_test,
        'predicted_price': predictions,
        'residual': y_test - predictions,
        'absolute_error': np.abs(y_test - predictions),
        'percentage_error': np.abs((y_test - predictions) / y_test) * 100
    })
    
    # Add prediction quality categories
    results_df['prediction_quality'] = pd.cut(
        results_df['percentage_error'], 
        bins=[0, 5, 10, 20, 100], 
        labels=['Excellent (<5%)', 'Good (5-10%)', 'Fair (10-20%)', 'Poor (>20%)']
    )
    
    predictions_path = os.path.join(artifacts_dir, "prediction_results.csv")
    results_df.to_csv(predictions_path, index=False)
    
    logger.info("✅ Created Detailed Predictions Analysis")
    return predictions_path

def create_model_summary_text(model, best_params, artifacts_dir="visualization_artifacts"):
    """
    Creates a human-readable model summary text file.
    """
    os.makedirs(artifacts_dir, exist_ok=True)
    
    summary_lines = [
        "="*60,
        "RANDOM FOREST REGRESSION MODEL SUMMARY",
        "="*60,
        "",
        f"Model Type: {type(model).__name__}",
        f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "OPTIMIZED HYPERPARAMETERS:",
        "-" * 30
    ]
    
    for param, value in best_params.items():
        summary_lines.append(f"{param}: {value}")
    
    summary_lines.extend([
        "",
        "MODEL CHARACTERISTICS:",
        "-" * 30,
        f"Number of Trees: {model.n_estimators}",
        f"Maximum Depth: {model.max_depth}",
        f"Minimum Samples Split: {model.min_samples_split}",
        f"Minimum Samples Leaf: {model.min_samples_leaf}",
        f"Random State: {model.random_state}",
        "",
        "FEATURE IMPORTANCE:",
        "-" * 30
    ])
    
    if hasattr(model, 'feature_importances_'):
        summary_lines.append("✅ Feature importance scores available for model interpretation")
    else:
        summary_lines.append("❌ Feature importance not available for this model type")
    
    summary_lines.extend([
        "",
        "USAGE NOTES:",
        "-" * 30,
        "• This model was trained using GridSearchCV with 3-fold cross-validation",
        "• Performance metrics include RMSE, MAE, R², MAPE, and Explained Variance",
        "• Model artifacts include visualizations and detailed analysis files",
        "• All hyperparameters were optimized for best MSE performance",
        "",
        "="*60
    ])
    
    summary_path = os.path.join(artifacts_dir, "model_summary.txt")
    with open(summary_path, 'w', encoding='utf-8') as f: # Added encoding='utf-8'
        f.write('\n'.join(summary_lines))
    
    logger.info("✅ Created Model Summary Text")
    return summary_path

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
    run_name = f"complete_artifacts_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    with mlflow.start_run(run_name=run_name) as run:
        logger.info(f"MLflow Run started with ID: {run.info.run_id}")
        logger.info(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")

        # Load the preprocessed data
        X_train, X_test, y_train, y_test = load_processed_data()
        
        # Perform Hyperparameter Tuning and get best model
        best_model, best_params, training_history = train_with_tuning(X_train, y_train)
        
        # Manual Logging of Parameters (Advanced Criteria)
        # Log the best hyperparameters found by GridSearchCV
        mlflow.log_params(best_params)
        logger.info("Manually logged best hyperparameters from tuning.")

        # Log data-related parameters for better traceability
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("test_samples", len(X_test))
        mlflow.log_param("features_count", X_train.shape[1])
        mlflow.log_param("cv_folds", 3)
        mlflow.log_param("param_combinations_tested", training_history['param_combinations'])
        logger.info("Manually logged data and training parameters.")

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

        # === CREATE AND LOG CUSTOM ARTIFACTS (Advanced Criteria) ===
        logger.info("🎨 Creating comprehensive visualizations and artifacts...")
        
        # Create regression visualizations (equivalent to confusion matrix for classification)
        visualization_paths = create_regression_visualizations(y_test, predictions, best_model, X_test)
        
        # Create training history visualization
        training_viz_path = create_training_history_visualization(training_history)
        visualization_paths.append(training_viz_path)
        
        # Create comprehensive model report
        model_report_path = create_comprehensive_model_report(
            best_model, best_params, rmse, mae, r2, additional_metrics, 
            training_history, X_train, X_test
        )
        
        # Create detailed predictions analysis
        predictions_analysis_path = save_predictions_and_analysis(y_test, predictions)
        
        # Create model summary text
        model_summary_path = create_model_summary_text(best_model, best_params)
        
        # Log all visualization artifacts
        for viz_path in visualization_paths:
            try:
                mlflow.log_artifact(viz_path, "visualizations")
                logger.info(f"✅ Logged visualization: {os.path.basename(viz_path)}")
            except Exception as e:
                logger.error(f"Failed to log visualization {viz_path}: {e}")
        
        # Log analysis artifacts
        analysis_artifacts = [model_report_path, predictions_analysis_path, model_summary_path]
        for artifact_path in analysis_artifacts:
            try:
                mlflow.log_artifact(artifact_path, "analysis")
                logger.info(f"✅ Logged analysis artifact: {os.path.basename(artifact_path)}")
            except Exception as e:
                logger.error(f"Failed to log analysis artifact {artifact_path}: {e}")

        # Save model locally first (as a backup)
        model_file_path = save_model_locally(best_model)
        
        # Manual Logging of the Model to DagsHub
        try:
            # Log the best model using mlflow.sklearn.log_model
            mlflow.sklearn.log_model(
                sk_model=best_model, 
                artifact_path="tuned_random_forest_model",
                input_example=X_train.iloc[:5] if hasattr(X_train, 'iloc') else X_train[:5],
                signature=mlflow.models.infer_signature(X_train, y_train)
            )
            logger.info("Successfully logged the best RandomForestRegressor model as an artifact to DagsHub.")
        except Exception as e:
            logger.error(f"Failed to log model to DagsHub via mlflow.sklearn.log_model: {e}")
            logger.warning("Attempting to log the locally saved model file as a generic artifact instead.")
            
            # Fallback: Log the locally saved .pkl file as a generic artifact
            try:
                mlflow.log_artifact(model_file_path, "model_backup")
                logger.info("Successfully logged local .pkl model file as artifact backup.")
            except Exception as backup_error:
                logger.error(f"Failed to log backup model artifact: {backup_error}")
        
        # --- Log additional run metadata ---
        try:
            # Get the experiment name
            current_experiment = mlflow.get_experiment(run.info.experiment_id)
            experiment_name = current_experiment.name if current_experiment else "Unknown"

            # Create a dictionary of key model and run information
            run_info_summary = {
                "Run ID": run.info.run_id,
                "Experiment Name": experiment_name, # Corrected
                "Model Type": "RandomForestRegressor",
                "Best Parameters": best_params,
                "Core Metrics": {"RMSE": rmse, "MAE": mae, "R2": r2},
                "Additional Metrics": additional_metrics,
                "Training Samples": len(X_train),
                "Test Samples": len(X_test),
                "Features Count": X_train.shape[1],
                "Timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "Artifacts Created": [
                    "actual_vs_predicted_plot.png",
                    "residuals_analysis_plot.png", 
                    "feature_importance_plot.png",
                    "performance_metrics_summary.png",
                    "training_history_visualization.png",
                    "comprehensive_model_report.json",
                    "prediction_results.csv",
                    "feature_importance_analysis.csv",
                    "model_summary.txt"
                ]
            }
            
            # Save this information to a file
            info_file_dir = "run_artifacts"
            os.makedirs(info_file_dir, exist_ok=True)
            info_file_path = os.path.join(info_file_dir, f"run_summary_{run.info.run_id}.txt")
            with open(info_file_path, 'w', encoding='utf-8') as f: # Added encoding='utf-8'
                f.write("="*60 + "\n")
                f.write("MLFLOW RUN SUMMARY\n")
                f.write("="*60 + "\n\n")
                for key, value in run_info_summary.items():
                    if key == "Artifacts Created":
                        f.write(f"{key}:\n")
                        for artifact in value:
                            f.write(f"  • {artifact}\n")
                    else:
                        f.write(f"{key}: {value}\n")
                f.write("\n" + "="*60 + "\n")
            
            # Log the info file as an artifact
            mlflow.log_artifact(info_file_path, "run_metadata")
            logger.info("Successfully logged run summary information as artifact.")
            
        except Exception as e:
            logger.warning(f"Failed to log run summary artifact: {e}")
        
        # --- Final summary output to console ---
        print(f"\n{'='*80}")
        print(f"🎉 TRAINING AND COMPREHENSIVE LOGGING COMPLETE")
        print(f"{'='*80}")
        print(f"MLflow Run ID: {run.info.run_id}")
        
        # Corrected experiment name retrieval for printing
        try:
            experiment_name_for_print = mlflow.get_experiment(run.info.experiment_id).name
            print(f"Experiment: {experiment_name_for_print}")
        except Exception:
            print(f"Experiment: Unknown (ID: {run.info.experiment_id})")

        print(f"\n📊 PERFORMANCE METRICS:")
        print(f"  • RMSE: {rmse:.2f}")
        print(f"  • MAE: {mae:.2f}")
        print(f"  • R² Score: {r2:.3f}")
        for name, value in additional_metrics.items():
            print(f"  • {name.upper()}: {value:.2f}")
        
        print(f"\n⚙️  BEST HYPERPARAMETERS:")
        for param, value in best_params.items():
            print(f"  • {param}: {value}")
        
        print(f"\n🎨 ARTIFACTS CREATED:")
        artifact_list = [
            "actual_vs_predicted_plot.png (Regression performance visualization)",
            "residuals_analysis_plot.png (Residuals distribution and patterns)",
            "feature_importance_plot.png (Top feature contributions)",
            "performance_metrics_summary.png (Metrics overview)",
            "training_history_visualization.png (Hyperparameter tuning results)",
            "comprehensive_model_report.json (Complete model documentation)",
            "prediction_results.csv (Detailed prediction analysis)",
            "feature_importance_analysis.csv (Feature importance data)",
            "model_summary.txt (Human-readable model description)"
        ]
        for artifact in artifact_list:
            print(f"  ✅ {artifact}")
        
        print(f"\n🔗 VIEW RESULTS:")
        print(f"  DagsHub MLflow UI: https://dagshub.com/nidaannisa06/membangun_model_advanced/mlflow")
        print(f"  Run ID: {run.info.run_id}")
        print(f"  Local Model: {model_file_path}")
        print(f"{'='*80}")

        # --- Write DagsHub URL to file ---
        try:
            with open("dagshub_mlflow_url.txt", "w", encoding='utf-8') as f: # Added encoding='utf-8'
                f.write(f"DagsHub MLflow Tracking URL\n")
                f.write(f"{'='*40}\n")
                f.write(f"Project URL: https://dagshub.com/nidaannisa06/membangun_model_advanced/mlflow\n")
                f.write(f"Run ID: {run.info.run_id}\n")
                f.write(f"Experiment ID: {run.info.experiment_id}\n")
                f.write(f"Run Name: {run_name}\n\n")
                f.write(f"Instructions:\n")
                f.write(f"1. Open the Project URL above\n")
                f.write(f"2. Look for the run with ID: {run.info.run_id}\n")
                f.write(f"3. Click on the run to view all metrics, parameters, and artifacts\n")
                f.write(f"4. Check the 'Artifacts' tab to see all generated visualizations and reports\n")
            logger.info("DagsHub MLflow URL and instructions written to dagshub_mlflow_url.txt")
        except Exception as e:
            logger.warning(f"Failed to write DagsHub URL file: {e}")

if __name__ == "__main__":
    main()
