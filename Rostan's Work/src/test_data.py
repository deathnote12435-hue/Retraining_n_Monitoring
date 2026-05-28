import pandas as pd 

#Load dataset 
df = pd.read_csv("../data/ObesityDataSet.csv")

#Show first 5 rows 
print(df.head())

#Show column names 
print(df.columns)

#Show dataset information
print(df.info())

#Show missing values
print(df.isnull().sum())