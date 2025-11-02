import os
import json
import tarfile
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

if __name__ == "__main__":
    # Extract the trained model from the model artifact tarball
    model_path = "/opt/ml/processing/model/model.tar.gz"
    with tarfile.open(model_path, 'r:gz') as tar:
        tar.extractall("/opt/ml/processing/model/")
    
    # Load the trained model using joblib
    model = joblib.load("/opt/ml/processing/model/model.joblib")
    
    # Load the test data prepared by the preprocessing step
    test_data_path = "/opt/ml/processing/test/test.csv"
    test_df = pd.read_csv(test_data_path)
    
    # Separate features and target variable
    X_test = test_df.drop("target", axis=1)

    y_test = test_df["target"]
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Create evaluation report
    evaluation_report = {
        "regression_metrics": {
            "mse": float(mse),
            "rmse": float(rmse),
            "mae": float(mae),
            "r2_score": float(r2)
        }
    }
    
    # Save the evaluation report as a JSON file in the expected output directory
    output_dir = "/opt/ml/processing/evaluation"
    os.makedirs(output_dir, exist_ok=True)
    
    with open(f"{output_dir}/evaluation.json", "w") as f:
        json.dump(evaluation_report, f)
    
    print("Evaluation completed!")