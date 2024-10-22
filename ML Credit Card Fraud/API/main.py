from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

# Load the XGBoost model pipeline using joblib
loaded_model = joblib.load("fraud_detection_model.pkl")

# Function to preprocess new data
def preprocess_data(transaction_dict):

    # Prepare the input data as a dictionary
    input_dict = {
        "merchant": transaction_dict["merchant"],
        "category": transaction_dict["category"],
        "amt": transaction_dict["amt"],
        "gender": transaction_dict["gender"],
        "zip": transaction_dict["zip"],
        "lat": transaction_dict["lat"],
        "long": transaction_dict["long"],
        "city_pop": transaction_dict["city_pop"],
        "merch_lat": transaction_dict["merch_lat"],
        "merch_long": transaction_dict["merch_long"],
        "transaction_hour": transaction_dict["transaction_hour"],
        "transaction_dayofweek": transaction_dict["transaction_dayofweek"],
        "transaction_month": transaction_dict["transaction_month"],
        "birth_year": transaction_dict["birth_year"],
        "birth_month": transaction_dict["birth_month"],
        "birth_day": transaction_dict["birth_day"]
    }

    # Convert to DataFrame
    transaction_df = pd.DataFrame([input_dict])

    return transaction_df

# Function to get predictions
def get_predictions(transaction_dict):
    preprocessed_data = preprocess_data(transaction_dict)
    prediction = loaded_model.predict(preprocessed_data)
    probability = loaded_model.predict_proba(preprocessed_data)
    return prediction, probability

@app.post("/predict")
async def predict(data: dict):

    # Make Prediction
    prediction, probabilities = get_predictions(data)

    return {
        "prediction": prediction.tolist(),
        "probabilities": probabilities.tolist()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
