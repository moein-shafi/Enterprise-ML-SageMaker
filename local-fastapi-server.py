import joblib
import pandas as pd

from fastapi import FastAPI
from fastapi import HTTPException, Request


model = joblib.load('trained_model.joblib')
print("Model loaded successfully.")

# Create FastAPI app
app = FastAPI(title="House Price Prediction API", version="1.0.0")


@app.post("/predict")
async def predict(request: Request):
    try:
        # Parse JSON body directly
        features = await request.json()
        # Convert to DataFrame
        input_data = pd.DataFrame([features])
        # Make prediction
        prediction = model.predict(input_data)[0]
        # Return a response
        return {
            "prediction": float(prediction),
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

