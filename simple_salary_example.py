
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


data = pd.read_csv('Position_Salaries.csv')
# print(data)

x = data.iloc[:, 1:2].values
y = data.iloc[:, 2].values

# choose number of trees here
regressor = RandomForestRegressor(n_estimators=100, random_state=0)

regressor.fit(x, y)

X_grid = np.arange(min(x), max(x), 0.01)
X_grid = X_grid.reshape(len(X_grid), 1)

plt.scatter(x, y, color='blue')

plt.plot(X_grid, regressor.predict(X_grid), color='green')
plt.title('Random Forest')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

y_pred = regressor.predict([[6.5]])
print('The predicted salary of a person at 6.5 level is $', y_pred[0])

