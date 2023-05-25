# snippets_sklearn
Repo to take notes about how to work with ML in sklearn

## Preparing data
Select X and y columns.

````python
# Input
X = df[['TotalSF']] # pandas DataFrame. Add '.values' to turn it into a Numpy array
# Label
y = df["SalePrice"] # pandas Series. Add '.values' to turn it into a Numpy array
````

## Train_test_split
Split data into train and test datasets

````python
# Import
from sklearn.model_selection import train_test_split
# Code
rain, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
````

## Scale the data
Specialy important if deep learning is going to be used. X_train and X_test get scaled, but not y.

````python
# Import
from sklearn.preprocessing import MinMaxScaler
# Code
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
````

## Linear Regression
````python
# Load the library
from sklearn.linear_model import LinearRegression
# Create an instance of the model
reg = LinearRegression()
# Fit the regressor
reg.fit(X,y)
# Do predictions
reg.predict([[2540],[3500],[4000]])
````
