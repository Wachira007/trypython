import pandas as pd
#saving the file path to a variable
n_file_path = 'd://train.csv'

#reading the file and storing it in a DATAFRAME
home_data = pd.read_csv(n_file_path)

#describe the data
a = home_data.describe()

#calculate the mean of lot size
lot_size = home_data['LotArea'].mean()

print(a)