import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sp
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
import warnings

#provides clean output 
warnings.filterwarnings("ignore")

#reads data from csv file 
df = pd.read_csv("Cars Dataset.csv")

#define the headers/columns of the data 
headers = ["Make", "Model", "Year", "Price", "Mileage"]
df.columns = headers
df.head()

#finding potential missing values 
data = df
#finding missing values
data.isna().any()
#finding if missing values
data.isnull().any()

#Changing the price which is a string into a interger
data.Price.unique()

#Here it contains '?', so we Drop it
data = data[data.Price != '?']

data['Price'] = data['Price'].astype(int)

x = np.array(df["Year"])
y = np.array(df["Mileage"])
z = np.array(df["Price"])

fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111, projection = '3d')
ax.scatter(x, y, z, c=z, cmap='viridis', marker='o')
ax.set_xlabel("Year")
ax.set_ylabel("Mileage")
ax.set_zlabel("Price")

plt.title("Example")
plt.show()
