## snippets_tensorflow

### Choosing optimizer and loss

#### For a multi_class classification problem
````py
model.compile(opimizer= 'rmsprop',
              loss= 'categorical_crossentropy',
              metrics= ['accuracy'])
````
#### For a binary classification problem
````py
model.compile(optimizer= 'rmsprop',
              loss= 'binary_crossentropy',
              metrics= ['accuracy'])
````
#### For a mean squared error regression problem
````py
model.compile(optimizer= 'rmsprop',
              loss= 'mse')
````

___
### Regression model
````py
# Imports
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_absolute_error, mean_squared_error, explained_variance_score
````

#### Build the model
````py
model = Sequential()
model.add(Dense(19, activation='relu')) # as many neurons per layer as features
model.add(Dense(19, activation='relu'))
model.add(Dense(19, activation='relu'))
model.add(Dense(19, activation='relu'))
model.add(Dense(1)) # output layer only 1 neuron
````

#### Compile the model
````py
model.compile(optimizer = 'adam', loss='mse')
````

#### Fit the model to train data with validation against test_data
````py
model.fit(x=X_train, y=y_train,
          validation_data=(X_test,y_test),
          batch_size = 128,
          epochs = 400)
````

#### Analize the model
````py
# Graphic analysis of loss drop
losses = pd.DataFrame(model.history.history)
losses.plot()
# MSE, MAE, Variance score
predictions = model.predict(X_test)
np.sqrt(mean_squared_error(y_test, predictions))
mean_absolute_error(y_test, predictions)
explained_variance_score(y_test, predictions)
# Graphic representation of linearity
plt.figure(figsize=(12,6))
plt.scatter(y_test, predictions)
plt.plot(y_test, y_test, 'r')
````
