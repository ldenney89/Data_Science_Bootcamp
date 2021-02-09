import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

regressor = DecisionTreeRegressor()
dataset = pd.read_csv('petrol_consumption.csv')


# independent variable
X = dataset['Temperature'].values
# dependnet variable
y = dataset['Revenue'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)
regressor.fit(X_train.reshape(-1, 1), y_train.reshape(-1, 1))

y_pred = regressor.predict(X_test.reshape(-1, 1))

df = pd.DataFrame({'Real Values': y_test.reshape(-1), 'Predicted Values': y_pred.reshape(-1)})

X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X_test, y_test, color='red')
plt.scatter(X_test, y_pred, color='green')
plt.title('Decision Tree Regression')
plt.xlabel('Temperature')
plt.ylabel('Revenue')
plt.show()

plt.plot(X_grid, regressor.predict(X_grid), color='black')
plt.title('Decision Tree Regression')
plt.xlabel('Temperature')
plt.ylabel('Revenue')
plt.show()
