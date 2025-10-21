import os
import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def model_fn(model_dir):
    """
    Load model for inference - SageMaker will handle the rest
    
    The model_fn function is a special function that SageMaker looks for when we later want to deploy our 
    model for making predictions. This function tells SageMaker how to load our saved model from the artifacts directory. 
    Even though we're not deploying the model in this lesson, including this function makes our training 
    script deployment-ready for future use. SageMaker will call this function automatically when setting up 
    inference endpoints, so it needs to know exactly how to reconstruct our trained model from the saved files.

    """
    # Load the saved model from the model directory using joblib
    # SageMaker expects the model to be saved as 'model.joblib'
    model = joblib.load(os.path.join(model_dir, 'model.joblib'))
    # Return the loaded model so SageMaker can use it for predictions
    return model


def train(args):
    """
    Train the model using data from the specified training directory.
    """
    # Load the training data from S3
    train_data_path = os.path.join(args.train, 'california_housing_train.csv')
    df = pd.read_csv(train_data_path)
    
    # Separate features and target variable
    X_train = df.drop("MedHouseVal", axis=1)  # Features
    y_train = df["MedHouseVal"]               # Target: median house value
    
    print(f"Starting training with {len(X_train)} examples...")
    print(f"Features: {list(X_train.columns)}")
    
    # Create and train the model directly
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Evaluate model performance on training data
    print(f"Starting evaluation...")
    y_train_pred = model.predict(X_train)
    train_r2 = r2_score(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    
    # Print training performance metrics
    print(f"R² Score: {train_r2:.4f}")
    print(f"RMSE: {train_rmse:.4f}")
    
    # Save the model for SageMaker deployment
    joblib.dump(model, os.path.join(args.model_dir, 'model.joblib'))
    print("Training completed!")


if __name__ == '__main__':
    """
    The if __name__ == '__main__': block is essential because SageMaker will import our script as a module when 
    it needs to use the model_fn function for inference. Without this guard, all our training code would execute 
    every time SageMaker imports the script, which isn't what we want. The guard ensures that the training logic 
    only runs when the script is executed directly during the training job.

    The key difference from standalone machine learning scripts is how it receives information about data locations 
    and where to save results. Instead of hardcoding file paths, the script uses argparse to read arguments that 
    SageMaker automatically provides through environment variables:

        SM_MODEL_DIR — SageMaker sets this environment variable to point to a directory (like /opt/ml/model/) on the 
        training instance where our script should save the trained model. After training completes, SageMaker automatically 
        uploads everything in this directory to S3 as model artifacts.

        SM_CHANNEL_TRAIN — SageMaker sets this to point to a directory (like /opt/ml/input/data/train/) on the training 
        instance where it has already downloaded our training data from S3. The "TRAIN" part must match the channel name 
        we'll specify later when we launch the training job.
    """


    # Create argument parser to handle SageMaker's environment variables
    parser = argparse.ArgumentParser()
    # Add argument for model directory - where SageMaker expects us to save the trained model
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    # Add argument for training data directory - where SageMaker downloads our training data
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    # Parse all the arguments that SageMaker provides
    args = parser.parse_args()
    print(f"Model directory: {args.model_dir}")
    print(f"Training data directory: {args.train}")

    # Start the training process
    train(args)
    