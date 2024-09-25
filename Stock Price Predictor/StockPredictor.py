import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

# Ignore warnings for clean output
import warnings
warnings.filterwarnings("ignore")

# Loads the historical stock data from a CSV file into a pandas DataFrame
data = pd.read_csv("sp500_stocks.csv")

#Gets a list of valid stock symbols from dataset
valid_symbols = data["Symbol"].unique()

#Ask what stock want to filter out 
while True:  
    name = input('What is the symbol of the stock you want to predict the price of?: ').strip().upper()
    if name in valid_symbols:
        break #exits the loop when valid symbol is entered
    else:
        print(f"{name} is not a valid symbol. Please try again.")

#Filters only that stocks data 
stock = data[data['Symbol'] == name]

# Extract the 'Close' price data for the stock to use for our predicition 
close_data = stock.filter(['Close'])
dataset = close_data.values  # Convert the 'Close' price column to a NumPy array for easier processing.

# Normalize the data (scaling) using MinMaxScaler to scale the 'Close' prices to a range between 0 and 1.
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

# Create training data 70% of the data to train the model and 30% for testing/prediction.
training_data_len = int(np.ceil(len(scaled_data) * 0.70))

# Create the training dataset & sequences of 60 days (x_train) & corresponding next day (y_train) to train the LSTM model.
train_data = scaled_data[0:int(training_data_len), :]
x_train = []
y_train = []

# Creates sequences of the previous 60 days of 'Close' prices to predict the next day's price.
for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])  # Last 60 days
    y_train.append(train_data[i, 0])  # The next day

# Convert training data into NumPy arrays, required by TensorFlow
x_train, y_train = np.array(x_train), np.array(y_train)

# Reshape x_train to a 3D array (samples, time steps, features) to fit LSTM input requirements.
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Build a basic LSTM neural network model with 2 LSTM layers, one Dense layer, and one Dropout layer to prevent overfitting.
model = keras.models.Sequential()

# First LSTM layer with 64 units and return_sequences=True since we are stacking another LSTM.
model.add(keras.layers.LSTM(units=64, return_sequences=True, input_shape=(x_train.shape[1], 1)))

# Second LSTM layer with 64 units.
model.add(keras.layers.LSTM(units=64))

# A Dense layer with 32 units (neurons) for further processing.
model.add(keras.layers.Dense(32))

# Dropout layer to prevent overfitting by ignoring 50% of the units during training.
model.add(keras.layers.Dropout(0.5))

# Final Dense layer to output the predicted stock price (only 1 value).
model.add(keras.layers.Dense(1))

# Compile the model using Adam optimizer and mean squared error as the loss function.
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model for 10 epochs on the x_train and y_train data.
history = model.fit(x_train, y_train, epochs=10)

# Prepare the test data for predictions using last 60 days from the training set to predict future prices.
test_data = scaled_data[training_data_len - 60:, :]

x_test = []
y_test = dataset[training_data_len:, :]  # The actual prices (not scaled) for comparison with predicted prices.

# Create sequences from the test data
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])

# Convert x_test to a NumPy array and reshape it to 3D (samples, time steps, features).
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Predict the stock prices for the test data
predictions = model.predict(x_test)

# The predictions are still in the scaled range (0 to 1), so we reverse the scaling.
predictions = scaler.inverse_transform(predictions)

# Evaluate the model by calculating the MSE between the predicted prices and the actual prices.
mse = np.mean(((predictions - y_test) ** 2))
print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {np.sqrt(mse)}")

# Visualize the predictions vs the actual stock prices 
train = stock[:training_data_len]
test = stock[training_data_len:]
test['Predictions'] = predictions  # Add the predicted prices to the test set.


# Plot the training and test data along with the predicted prices.
plt.figure(figsize=(10, 8))
plt.plot(train['Date'], train['Close'], label="Train Data")  # Plot the training data
plt.plot(test['Date'], test['Close'], label="Actual Prices")  # Plot the actual prices
plt.plot(test['Date'], test['Predictions'], label="Predicted Prices")  # Plot the predicted prices
plt.title(f"{name} Stock Price Prediction")
plt.xlabel('Date')
plt.ylabel('Close Price (USD)')
plt.legend()
plt.show()

# Print the training data (dates and close prices)
print("Training Data:")
print(train[['Date', 'Close']])  

# Print the actual prices and predicted prices for the test data (dates, close prices, and predictions)
print("Test Data with Actual and Predicted Prices:")
print(test[['Date', 'Close', 'Predictions']])  

# Save the training data to a CSV file
train[['Date', 'Close']].to_csv(f"{name}_training_data.csv", index=False)
print(f"Training data saved to {name}_training_data.csv")

# Save the test data along with actual and predicted prices to a CSV file
test[['Date', 'Close', 'Predictions']].to_csv(f"{name}_test_predictions.csv", index=False)
print(f"Test data with actual and predicted prices saved to {name}_test_predictions.csv")

#Predict stock prices for the next 30 days after 9/13/2024.

# Get the last 60 days of scaled data to make future predictions
last_60_days = scaled_data[-60:]
x_future = [last_60_days]  # Prepare a sequence of the last 60 days
x_future = np.array(x_future)
x_future = np.reshape(x_future, (x_future.shape[0], x_future.shape[1], 1))

# Predict for the next 30 business days
future_predictions = []
for _ in range(30):  # Predict the next 30 days
    pred_price = model.predict(x_future)  # Predict the next day's price
    future_predictions.append(pred_price[0, 0])  # Store the prediction

    # Add the prediction to the input and remove the oldest day (sliding window approach)
    x_future = np.append(x_future[0][1:], pred_price, axis=0)
    x_future = np.reshape(x_future, (1, x_future.shape[0], 1))

# Inverse scale the predictions back to the original price range
future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

# Create a list of future dates for plotting
future_dates = pd.date_range(start='2024-09-14', periods=len(future_predictions), freq='B')  # 'B' stands for business days

# Plot the future predictions
plt.figure(figsize=(10, 8))
plt.plot(future_dates, future_predictions, label='Predicted Prices')
plt.title(f"{name} Stock Price Predictions After 9/13/2024")
plt.xlabel('Date')
plt.ylabel('Predicted Close Price (USD)')
plt.legend()
plt.show()

# Print future dates and predicted prices
print("Future Dates and Predicted Prices:")
for date, prediction in zip(future_dates, future_predictions):
    print(f"Date: {date}, Predicted Price: {prediction[0]}")

# Create a DataFrame for future dates and predictions
future_df = pd.DataFrame({
    'Date': future_dates,
    'Predicted Price': [pred[0] for pred in future_predictions]  # Extract the values from the predictions array
})

# Save the future predictions to a CSV file
future_df.to_csv(f"{name}_future_predictions.csv", index=False)
print(f"Future predictions saved to {name}_future_predictions.csv")
