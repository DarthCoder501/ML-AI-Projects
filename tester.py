# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from pandas.plotting import parallel_coordinates
import warnings

# Filter warnings for cleaner output
warnings.filterwarnings("ignore")

# Loads and cleans the data 
def load_and_clean_data(file_path):
    df = pd.read_csv(file_path)
    headers = ["Make", "Model", "Year", "Price", "Mileage"]
    df.columns = headers
    
    # Remove rows with invalid price data
    df = df[df.Price != '?']
    # Turns prices that are strs into ints
    df['Price'] = df['Price'].astype(int)
    
    return df

# Visualize Data Distribution
def visualize_data_distribution(df):
    plt.figure(figsize=(15,10))
    sns.histplot(df["Price"], kde=True)
    plt.title("Distribution of Car Prices")
    plt.xlabel("Car Price ($)")
    plt.ylabel("Number of Cars")
    plt.show()

# Train Random Forest Regressor Model
def train_model(df):
    X = df[["Year", "Mileage"]]
    y = df["Price"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
    # Apply Random Forest Regressor
    regressor = RandomForestRegressor(n_estimators=100, random_state=0)
    regressor.fit(X_train, y_train)
    
    y_pred = regressor.predict(X_test)
    
    return X_test, y_test, y_pred, regressor

# Visualize Actual vs Predicted Prices
def visualize_predictions(y_test, y_pred):
    comparison_df = pd.DataFrame({'Actual Price': y_test.values, 'Predicted Price': y_pred})
    comparison_df.reset_index(inplace=True)
    
    plt.figure(figsize=(12, 8))
    plt.plot(comparison_df.index, comparison_df['Actual Price'], label='Actual Price', marker='o')
    plt.plot(comparison_df.index, comparison_df['Predicted Price'], label='Predicted Price', marker='x')
    plt.xlabel("Sample Index")
    plt.ylabel("Price ($)")
    plt.title("Comparison of Actual vs Predicted Car Prices")
    plt.legend()
    plt.grid(True)
    plt.show()

# Show Model Error Metrics
def print_model_metrics(y_test, y_pred):
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
    print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

# 3D Scatter Plot of Car Data
def plot_3d_scatter(df):
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')

    unique_makes = df['Make'].unique()
    marker_styles = ['o', '^', 's', 'D', 'v', 'P', 'X', '*', '+', 'x']
    
    # Generate a colormap with enough colors for all makes
    colormap = cm.get_cmap('tab20', len(unique_makes))

    # Ensure enough markers or throw an error
    if len(unique_makes) > len(marker_styles):
        print("Not enough unique markers. Using colors to differentiate makes.")

    for i, make in enumerate(unique_makes):
        make_mask = df['Make'] == make
        ax.scatter(df['Year'][make_mask], df['Mileage'][make_mask], df['Price'][make_mask],
                   marker=marker_styles[i % len(marker_styles)], # Reuse markers if needed
                   color=colormap(i), label=make, s=50)

    ax.set_xlabel("Car Year")
    ax.set_ylabel("Car Mileage (miles)")
    ax.set_zlabel("Car Price ($)")
    plt.title("Car Prices by Year, Mileage, and Make")
    ax.legend(title="Make")
    plt.show()

# Plot Regression Surface
def plot_regression_surface(df, model):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    # Define a grid over the Year and Mileage
    year_range = np.linspace(df['Year'].min(), df['Year'].max(), 50)
    mileage_range = np.linspace(df['Mileage'].min(), df['Mileage'].max(), 50)
    year_grid, mileage_grid = np.meshgrid(year_range, mileage_range)
    
    # Create a flattened version of the grid for prediction
    iv_grid = np.c_[year_grid.ravel(), mileage_grid.ravel()]
    
    # Predict the car prices using the Random Forest model
    predicted_prices = model.predict(iv_grid)
    
    # Reshape the predicted prices back into the shape of the grid
    predicted_prices = predicted_prices.reshape(year_grid.shape)
    
    # Plot the actual data points
    ax.scatter(df['Year'], df['Mileage'], df['Price'], color='blue', label='Actual Data', s=20)
    
    # Plot the regression surface
    ax.plot_surface(year_grid, mileage_grid, predicted_prices, color='red', alpha=0.5)
    
    # Axis labels
    ax.set_xlabel('Car Year')
    ax.set_ylabel('Car Mileage (miles)')
    ax.set_zlabel('Estimated Price ($)')
    
    plt.title('3D Regression Surface of Car Prices')
    plt.show()

# Correlation Heatmap
def plot_correlation_heatmap(df):
    corr_matrix = df[['Year', 'Mileage', 'Price']].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.title("Correlation Matrix for Car Price Factors")
    plt.show()

# Function to make predictions with new data
def make_predictions(model):
    while True: 
        yr = input("What year is the car?: ").strip()
        if yr.isnumeric() == True: 
            yr = int(yr)
            break 
        else: 
            print("Please enter a valid year")

    while True: 
        miles = input("How many miles does the car have?: ").strip()
        if miles.isnumeric() == True: 
            miles = int(miles)
            break 
        else: 
            print("Please enter a valid year")

    new_data = pd.DataFrame({'Year': [yr], 'Mileage': [miles]})
    predicted_price = model.predict(new_data)
    print(f"The predicted price for a {yr} car with {miles} miles is: ${predicted_price[0]:,.2f}")

# Main Function to Run All Steps
def main():
    # Load and clean data
    df = load_and_clean_data("Cars Dataset.csv")
    
    # Visualize price distribution
    visualize_data_distribution(df)
    
    # Train the model and visualize predictions
    X_test, y_test, y_pred, model = train_model(df)
    visualize_predictions(y_test, y_pred)
    
    # Print error metrics
    print_model_metrics(y_test, y_pred)
    
    # Plot 3D scatter plot
    plot_3d_scatter(df)
    
    # Plot regression surface
    plot_regression_surface(df, model)
    
    # Correlation heatmap
    plot_correlation_heatmap(df)

    # Make predictions with new data
    make_predictions(model)

# Run the main function
if __name__ == "__main__":
    main()
