import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.api.types import is_numeric_dtype, is_string_dtype

df = pd.read_csv('weatherAUS.csv')

df.describe()
df.isnull().sum()   # num of nulls
df.isnull().count()   # same total value for each row
round(df.isnull().sum()/df.isnull().count() * 100, 1)   # percent missing of total counts

# drop columns Evaporation, Sunshine, Cloud9am, and Cloud3pm bc they have almost 40% of the data missing
df.drop(['Evaporation','Sunshine','Cloud9am','Cloud3pm'], axis = 1, inplace = True)

# drop rows with missing values in RainTomorrow column bc we need values to test our model with
df.dropna(subset = ['RainTomorrow'], inplace = True)

# classify columns as numerical or categorical
numVar, catVar = [], []
for column in df:
    if column != 'RainTomorrow':
        if is_numeric_dtype(df[column]):
            numVar.append(column)
        elif is_string_dtype(df[column]):
            catVar.append(column)
            
# fill missing values in numerical columns with means of the column
df.fillna(df.mean(), inplace = True)

# fill missing values in categorical columns with "Unknown"
for i in catVar:
    if df[i].isnull().any():
        df[i].fillna('Unknown', inplace = True)
        
# df_out = df.to_csv('aus_rain.csv', index = False)

# remove the outliers in the Rainfall column that we noticed in our EDA
df = df[df['Rainfall'] < df['Rainfall'].quantile(.9)]

# Feature Engineering


        