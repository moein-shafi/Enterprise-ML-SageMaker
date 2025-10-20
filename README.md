# Enterprise-ML-SageMaker

## Setup & requirements — Amazon SageMaker

Follow these steps to prepare a local environment for working with the Amazon SageMaker Python SDK and the helper scripts in this repo.

1. Prerequisites
   - An AWS account.
   - An IAM user with permissions to use SageMaker and S3 (for quickstart you can attach `AmazonSageMakerFullAccess` and `AmazonS3FullAccess`; for production use apply least-privilege policies).
   - Install Python 3.8+ on Windows and ensure `python` is on your PATH.

2. Install and configure AWS CLI
   - Install AWS CLI v2 for Windows (use the official installer from AWS).
   - Configure credentials:
     - Interactive: `aws configure` and enter AWS Access Key ID, Secret, and region.
     - Or set environment variables:
       - PowerShell:
         ```
         setx AWS_ACCESS_KEY_ID "YOUR_KEY"
         setx AWS_SECRET_ACCESS_KEY "YOUR_SECRET"
         setx AWS_DEFAULT_REGION "us-west-2"
         ```
       - CMD:
         ```
         set AWS_ACCESS_KEY_ID=YOUR_KEY
         set AWS_SECRET_ACCESS_KEY=YOUR_SECRET
         set AWS_DEFAULT_REGION=us-west-2
         ```

3. Create a Python virtual environment (Windows)
   - PowerShell:
     ```
     python -m venv .venv
     .\.venv\Scripts\Activate.ps1
     ```
   - CMD:
     ```
     python -m venv .venv
     .\.venv\Scripts\activate.bat
     ```

4. Install Python dependencies
   - Create a `requirements.txt` (example provided below) or install packages directly:
     ```
     python -m pip install --upgrade pip
     pip install -r requirements.txt
     ```
   - Example minimal `requirements.txt`:
     ```text
     sagemaker>=2.0.0
     boto3
     awscli
     pandas
     scikit-learn
     requests
     ```

5. Verify the setup
   - Quick Python check (prints the SageMaker default bucket — ensure credentials are configured):
     ```
     python -c "import sagemaker; print(sagemaker.Session().default_bucket())"
     ```
   - Run provided helper (example):
     ```
     python s3-management.py
     ```

Notes
- For production or CI, prefer using IAM roles attached to the environment (EC2, EKS, or SageMaker notebooks) instead of long-lived credentials.
- Adjust package versions in `requirements.txt` to match your organizational compatibility policy.