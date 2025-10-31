import os
import pandas as pd
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    # Read input data from SageMaker's input directory
    input_data_path = "/opt/ml/processing/input/california_housing.csv"
    df = pd.read_csv(input_data_path)
    
    print(f"Processing {len(df)} samples...")
    
    # Apply the same preprocessing steps you've done before:
    # 1. Cap outliers at 95th percentiles (excluding geographic coordinates)
    # 2. Create RoomsPerHousehold feature
    # 3. Select relevant features for modeling
    # ... (preprocessing logic here) ...
    # Cap outliers
    for col in df.columns:
        if col not in ["Latitude", "Longitude"]:
            cap_value = df[col].quantile(0.95)
            df[col] = df[col].clip(upper=cap_value)
    # Create new feature
    df["RoomsPerHousehold"] = df["TotalRooms"] / df["Households"]
    # Select relevant features  
    feature_cols = [
        "MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population",
        "AveOccup", "Latitude", "Longitude", "RoomsPerHousehold"
    ]
    X = df[feature_cols]
    y = df["MedHouseVal"]
    
    
    # Split the data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create output directories for SageMaker
    os.makedirs("/opt/ml/processing/train", exist_ok=True)
    os.makedirs("/opt/ml/processing/test", exist_ok=True)
    
    # Combine features and target for saving
    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)
    
    # Save to SageMaker's expected output directories
    train_data.to_csv("/opt/ml/processing/train/train.csv", index=False)
    test_data.to_csv("/opt/ml/processing/test/test.csv", index=False)
    
    print("Data processing completed!")
    print(f"Train samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")