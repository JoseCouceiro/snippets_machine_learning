# snippets_sklearn
Repo to take notes about how to work with ML in sklearn

## Preparing data
````python
# Input
X = df[['TotalSF']] # pandas DataFrame
# Label
y = df["SalePrice"] # pandas Series
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
