import sagemaker


def upload_data_to_s3(sagemaker_session, default_bucket, TRAIN_DATA_PATH, DATA_PREFIX):
    try:
        # Upload the training data using SageMaker session
        train_s3_uri = sagemaker_session.upload_data(
            path=TRAIN_DATA_PATH,        # Local file to upload
            bucket=default_bucket,       # S3 bucket destination
            key_prefix=DATA_PREFIX       # Folder structure in S3
        )
        
        # Print success message
        print(f"Successfully uploaded training data to {train_s3_uri}")
        
    except Exception as e:
        print(f"Error: {e}")


def download_data_from_s3(sagemaker_session, default_bucket, DATA_PREFIX):
    try:
        # Download the data using SageMaker session
        sagemaker_session.download_data(
            path=DATA_PREFIX,  # Local destination path
            bucket=default_bucket,   # S3 bucket to download from
            key_prefix=f"{DATA_PREFIX}/california_housing_train.csv"  # S3 object key (file path in bucket)
        )

        # Print success message
        print(f"Successfully downloaded data to {DATA_PREFIX}/california_housing_train.csv")
        
    except Exception as e:
        print(f"Error: {e}")


def main():
    # Initialize SageMaker session
    sagemaker_session = sagemaker.Session()

    # Get the default SageMaker bucket name
    default_bucket = sagemaker_session.default_bucket()

    # Print the bucket name to see what it looks like
    print(f"Default SageMaker bucket: {default_bucket}")

    # Local file path
    TRAIN_DATA_PATH = "data/california_housing_train.csv"

    # S3 prefix (folder path within the bucket)
    DATA_PREFIX = "datasets"
    # Upload data to S3
    upload_data_to_s3(sagemaker_session, default_bucket, TRAIN_DATA_PATH, DATA_PREFIX)
    # Download data from S3
    download_data_from_s3(sagemaker_session, default_bucket, DATA_PREFIX)


if __name__ == "__main__":
    main()