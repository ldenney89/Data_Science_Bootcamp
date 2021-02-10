# -------------------------------------------------#
# Title: Petrol Consumption
# Dev:   LDenney
# Date:  February 9th, 2021
# ChangeLog: (Who, When, What)
#   Laura Denney, 2/9/21 Created File
# -------------------------------------------------#


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

regressor = DecisionTreeRegressor()
dataset = pd.read_csv('petrol_consumption.csv')

'''
# independent variables
x_tax = dataset['Petrol_tax'].values
x_income = dataset['Average_income'].values
x_highway = dataset['Paved_Highways'].values
x_population = dataset['Population_Driver_licence(%)'].values
'''
# dependent variable
y = dataset['Petrol_Consumption'].values
#independent variables
x_values = ['Petrol_tax', 'Average_income', 'Paved_Highways', 'Population_Driver_licence(%)']

###############################################
for item in x_values:
    x = dataset[item].values
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    regressor.fit(x_train.reshape(-1, 1), y_train.reshape(-1, 1))

    y_pred = regressor.predict(x_test.reshape(-1, 1))

    df = pd.DataFrame({'Real Values': y_test.reshape(-1), 'Predicted Values': y_pred.reshape(-1)})

    X_grid = np.arange(min(x), max(x), 0.01)
    X_grid = X_grid.reshape((len(X_grid), 1))
    plt.scatter(x_test, y_test, color='red')
    plt.scatter(x_test, y_pred, color='green')
    plt.title('Decision Tree Regression')
    plt.xlabel(item)
    plt.ylabel('Petrol Consumption')
    plt.show()

    plt.plot(X_grid, regressor.predict(X_grid), color='black')
    plt.title('Decision Tree Regression')
    plt.xlabel(item)
    plt.ylabel('Petrol Consumption')
    plt.show()

'''
###############################################
#Petrol Tax
x_tax_train, x_tax_test, y_tax_train, y_tax_test = train_test_split(x_tax, y, test_size=0.3)
regressor.fit(x_tax_train.reshape(-1, 1), y_tax_train.reshape(-1, 1))

y_tax_pred = regressor.predict(x_tax_test.reshape(-1, 1))

df = pd.DataFrame({'Real Values': y_tax_test.reshape(-1), 'Predicted Values': y_tax_pred.reshape(-1)})

X_grid = np.arange(min(x_tax), max(x_tax), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(x_tax_test, y_tax_test, color='red')
plt.scatter(x_tax_test, y_tax_pred, color='green')
plt.title('Decision Tree Regression')
plt.xlabel('Petrol Tax')
plt.ylabel('Petrol Consumption')
plt.show()

plt.plot(X_grid, regressor.predict(X_grid), color='black')
plt.title('Decision Tree Regression')
plt.xlabel('Petrol Tax')
plt.ylabel('Petrol Consumption')
plt.show()

###############################################
#Average Income
x_income_train, x_income_test, y_income_train, y_income_test = train_test_split(x_income, y, test_size=0.3)
regressor.fit(x_income_train.reshape(-1, 1), y_income_train.reshape(-1, 1))

y_income_pred = regressor.predict(x_income_test.reshape(-1, 1))

df = pd.DataFrame({'Real Values': y_income_test.reshape(-1), 'Predicted Values': y_income_pred.reshape(-1)})

X_grid = np.arange(min(x_income), max(x_income), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(x_income_test, y_income_test, color='red')
plt.scatter(x_income_test, y_income_pred, color='green')
plt.title('Decision Tree Regression')
plt.xlabel('Average Income')
plt.ylabel('Petrol Consumption')
plt.show()

plt.plot(X_grid, regressor.predict(X_grid), color='black')
plt.title('Decision Tree Regression')
plt.xlabel('Average Income')
plt.ylabel('Petrol Consumption')
plt.show()

###############################################
#Paved Highways
x_highway_train, x_highway_test, y_highway_train, y_highway_test = train_test_split(x_highway, y, test_size=0.3)
regressor.fit(x_highway_train.reshape(-1, 1), y_highway_train.reshape(-1, 1))

y_highway_pred = regressor.predict(x_highway_test.reshape(-1, 1))

df = pd.DataFrame({'Real Values': y_highway_test.reshape(-1), 'Predicted Values': y_highway_pred.reshape(-1)})

X_grid = np.arange(min(x_highway), max(x_highway), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(x_highway_test, y_highway_test, color='red')
plt.scatter(x_highway_test, y_highway_pred, color='green')
plt.title('Decision Tree Regression')
plt.xlabel('Paved Highway')
plt.ylabel('Petrol Consumption')
plt.show()

plt.plot(X_grid, regressor.predict(X_grid), color='black')
plt.title('Decision Tree Regression')
plt.xlabel('Paved Highway')
plt.ylabel('Petrol Consumption')
plt.show()

###############################################
#Population of Driver's License
x_population_train, x_population_test, y_population_train, y_population_test = train_test_split(x_population, y, test_size=0.3)
regressor.fit(x_population_train.reshape(-1, 1), y_population_train.reshape(-1, 1))

y_population_pred = regressor.predict(x_population_test.reshape(-1, 1))

df = pd.DataFrame({'Real Values': y_population_test.reshape(-1), 'Predicted Values': y_population_pred.reshape(-1)})

X_grid = np.arange(min(x_population), max(x_population), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(x_population_test, y_population_test, color='red')
plt.scatter(x_population_test, y_population_pred, color='green')
plt.title('Decision Tree Regression')
plt.xlabel('Population of Driver License Percentage')
plt.ylabel('Petrol Consumption')
plt.show()

plt.plot(X_grid, regressor.predict(X_grid), color='black')
plt.title('Decision Tree Regression')
plt.xlabel('Population of Driver License Percentage')
plt.ylabel('Petrol Consumption')
plt.show()
'''