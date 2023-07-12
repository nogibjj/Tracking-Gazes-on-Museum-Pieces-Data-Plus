import pandas as pd
dataframe = pd.read_csv("2021_1a_tracking_data.txt",delimiter=r"\t")
print(dataframe.head())
dataframe.to_csv('output.csv')