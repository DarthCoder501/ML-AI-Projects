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
data = pd.read_csv("Cars Dataset.csv")

#removing index column 
data = data.iloc[: , 1:]

#define the headers/columns of the data 
headers = ["Make", "Model", "Year", "Price", "Mileage"]
data.columns = headers
data.head()

#finding potential missing values 
df = data
#finding missing values
df.isna().any()
#finding if missing values
df.isnull().any()

print(df.columns)
data.dtypes
