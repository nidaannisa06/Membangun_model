import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# MLflow Setup
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Housing Price Prediction - Complete Artifacts")
logger.info(f"MLflow Tracking configured to: {mlflow.get_tracking_uri()}")

def load_processed_data(data_path="housing_preprocessing"):
    """Load processed data from CSV files."""
    try:
        X_train = pd.read_csv(os.path.join(data_path, "X_train.csv"))
        X_test = pd.read_csv(os.path.join(data_path, "X_test.csv"))
        y_train = pd.read_csv(os.path.join(data_path, "y_train.csv"))['price']
        y_test = pd.read_csv(os.path.join(data_path, "y_test.csv"))['price']

        logger.info("✅ Data loaded successfully.")
        logger.info(f"📊 X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logger.error(f"❌ Error loading processed data: {str(e)}")
        raise

def create_visualizations(model, X_train, X_test, y_test, predictions):
    """Create and save visualization artifacts."""
    logger.info("Creating visualization artifacts...")
    
    # 1. Feature Importance Plot
    plt.figure(figsize=(12, 8))
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False).head(15)
    
    sns.barplot(data=feature_importance, y='feature', x='importance', palette='viridis')
    plt.title('Top 15 Feature Importances', fontsize=16, fontweight='bold')
    plt.xlabel('Importance Score', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.tight_layout()
    plt.savefig('feature_importance_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Actual vs Predicted Scatter Plot
    plt.figure(figsize=(10, 8))
    plt.scatter(y_test, predictions, alpha=0.6, color='blue', s=50)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Values', fontsize=12)
    plt.ylabel('Predicted Values', fontsize=12)
    plt.title('Actual vs Predicted Values', fontsize=16, fontweight='bold')
    
    # Add R² score to the plot
    r2 = r2_score(y_test, predictions)
    plt.text(0.05, 0.95, f'R² = {r2:.3f}', transform=plt.gca().transAxes, 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.tight_layout()
    plt.savefig('actual_vs_predicted_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Residual Plot
    plt.figure(figsize=(10, 6))
    residuals = y_test - predictions
    plt.scatter(predictions, residuals, alpha=0.6, color='green')
    plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
    plt.xlabel('Predicted Values', fontsize=12)
    plt.ylabel('Residuals', fontsize=12)
    plt.title('Residual Plot', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('residual_analysis_plot.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Distribution of Residuals
    plt.figure(figsize=(10, 6))
    plt.hist(residuals, bins=50, alpha=0.7, color='purple', edgecolor='black')
    plt.xlabel('Residuals', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Distribution of Residuals', fontsize=16, fontweight='bold')
    plt.axvline(x=0, color='red', linestyle='--', linewidth=2)
    plt.tight_layout()
    plt.savefig('residual_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("✅ Visualization artifacts created successfully.")
    return feature_importance

def save_analysis_files(model, X_train, X_test, y_test, predictions, feature_importance):
    """Save analysis files as artifacts."""
    logger.info("Creating analysis file artifacts...")
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    # 1. Model Performance Report
    performance_report = {
        "model_performance": {
            "rmse": float(rmse),
            "mae": float(mae),
            "r2_score": float(r2),
            "mse": float(mean_squared_error(y_test, predictions))
        },
        "dataset_info": {
            "n_features": len(X_train.columns),
            "n_train_samples": len(X_train),
            "n_test_samples": len(X_test),
            "feature_names": list(X_train.columns)
        },
        "model_config": {
            "model_type": "RandomForestRegressor",
            "n_estimators": model.n_estimators,
            "max_depth": model.max_depth,
            "min_samples_split": model.min_samples_split,
            "min_samples_leaf": model.min_samples_leaf,
            "random_state": model.random_state
        }
    }
    
    with open('comprehensive_model_report.json', 'w') as f:
        json.dump(performance_report, f, indent=2)
    
    # 2. Feature Importance CSV
    feature_importance.to_csv('feature_importance_analysis.csv', index=False)
    
    # 3. Predictions vs Actual CSV
    results_df = pd.DataFrame({
        'actual': y_test.values,
        'predicted': predictions,
        'residuals': y_test.values - predictions,
        'absolute_error': np.abs(y_test.values - predictions)
    })
    results_df.to_csv('prediction_results.csv', index=False)
    
    # 4. Model Summary Text
    with open('model_summary.txt', 'w') as f:
        f.write("=== HOUSING PRICE PREDICTION MODEL ===\n\n")
        f.write(f"Model Type: Random Forest Regressor\n")
        f.write(f"Training Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("=== PERFORMANCE METRICS ===\n")
        f.write(f"RMSE: {rmse:.2f}\n")
        f.write(f"MAE: {mae:.2f}\n")
        f.write(f"R² Score: {r2:.4f}\n\n")
        f.write("=== DATASET INFO ===\n")
        f.write(f"Training Samples: {len(X_train)}\n")
        f.write(f"Test Samples: {len(X_test)}\n")
        f.write(f"Number of Features: {len(X_train.columns)}\n\n")
        f.write("=== TOP 10 MOST IMPORTANT FEATURES ===\n")
        for idx, row in feature_importance.head(10).iterrows():
            f.write(f"{row['feature']}: {row['importance']:.4f}\n")
    
    logger.info("✅ Analysis file artifacts created successfully.")

def main():
    # *** KUNCI: AKTIFKAN AUTOLOG PENUH TANPA KONFLIK ***
    mlflow.sklearn.autolog(
        log_input_examples=True,
        log_model_signatures=True, 
        log_models=True,  # ✅ PASTIKAN INI TRUE
        disable=False,
        exclusive=False,
        disable_for_unsupported_versions=False,
        silent=False,
        log_post_training_metrics=True
    )
    
    logger.info("🚀 MLflow autolog FULLY ENABLED - akan menghasilkan model artifacts!")
    
    with mlflow.start_run() as run:
        logger.info(f"MLflow Run started with ID: {run.info.run_id}")
        
        # Load data
        X_train, X_test, y_train, y_test = load_processed_data()
        
        # *** TRAINING MODEL - AUTOLOG AKAN MENANGKAP SEMUA ***
        logger.info("🔄 Training model - autolog akan mengcapture parameters, metrics, dan model...")
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
        
        # FIT MODEL - autolog akan bekerja di sini
        model.fit(X_train, y_train)
        logger.info("✅ Model training complete - autolog captured parameters!")
        
        # PREDICT - autolog akan capture metrics
        predictions = model.predict(X_test)
        
        # EVALUASI - autolog akan capture metrics ini
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        logger.info(f"📊 Metrics - RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.4f}")
        
        # *** TAMBAHAN: LOG MODEL SECARA EKSPLISIT JUGA ***
        # Ini untuk memastikan model artifacts muncul
        try:
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="trained_model",
                input_example=X_train.iloc[:3],
                signature=mlflow.models.infer_signature(X_train, y_train)
            )
            logger.info("✅ Additional explicit model logging successful!")
        except Exception as e:
            logger.warning(f"⚠️ Explicit model logging failed: {e}")
        
        # Create additional artifacts
        feature_importance = create_visualizations(model, X_train, X_test, y_test, predictions)
        save_analysis_files(model, X_train, X_test, y_test, predictions, feature_importance)
        
        # Log all artifacts
        mlflow.log_artifact('feature_importance_plot.png')
        mlflow.log_artifact('actual_vs_predicted_plot.png') 
        mlflow.log_artifact('residual_analysis_plot.png')
        mlflow.log_artifact('residual_distribution.png')
        mlflow.log_artifact('comprehensive_model_report.json')
        mlflow.log_artifact('feature_importance_analysis.csv')
        mlflow.log_artifact('prediction_results.csv')
        mlflow.log_artifact('model_summary.txt')
        
        # Log additional custom metrics
        mlflow.log_metric("custom_rmse", rmse)
        mlflow.log_metric("custom_mae", mae)
        mlflow.log_metric("custom_r2", r2)
        mlflow.log_metric("n_features", len(X_train.columns))
        
        # Log custom parameters  
        mlflow.log_param("data_source", "housing_preprocessing")
        mlflow.log_param("model_version", "v2.0")
        mlflow.log_param("training_samples", len(X_train))
        mlflow.log_param("test_samples", len(X_test))
        
        logger.info("="*60)
        logger.info("🎉 TRAINING COMPLETED SUCCESSFULLY!")
        logger.info(f"📈 Final Performance - RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.4f}")
        logger.info(f"🔗 MLflow UI: {mlflow.get_tracking_uri()}/#/experiments")
        logger.info("📁 Expected Artifacts in MLflow UI:")
        logger.info("   ✅ model/ (from autolog)")
        logger.info("   ✅ trained_model/ (explicit logging)")
        logger.info("   ✅ Feature importance plots & analysis")
        logger.info("   ✅ Residual analysis & predictions")
        logger.info("   ✅ Comprehensive reports & summaries")
        logger.info("🔍 Check Artifacts tab in MLflow UI untuk melihat semua file!")
        logger.info("="*60)

if __name__ == "__main__":
    main()