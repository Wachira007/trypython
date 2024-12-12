import pandas as pd
from sklearn.tree import DecisionTreeRegressor 
#saving the file path to a variable
n_file_path = 'd://train.csv'

#reading the file and storing it in a DATAFRAME
home_data = pd.read_csv(n_file_path)

#describe the data
a = home_data.describe()


#calculate the mean of lot size
lot_size = home_data['LotArea'].mean()

#calculate the number of houses
number_of_houses = home_data.shape[0]

home_data.columns
print(number_of_houses)

#selecting the target variable
y = home_data.SalePrice

#selecting the features
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = home_data[features]

#review the data
X.describe() #summary of the data
X.head() #First 5 rows of the data
#defining the model
nairobi_model = DecisionTreeRegressor(random_state=1)

#fitting the model
nairobi_model.fit(X, y)

#predicting the price of the first 5 houses
print("The predictions are")
print(X.head())
print(nairobi_model.predict(X.head()))
print("The actual prices are")
print(y.head())
