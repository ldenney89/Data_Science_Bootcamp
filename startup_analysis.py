import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns


data = pd.read_csv("50_Startups.csv")

data.info()
print()
print(data.describe())
print()

features = data.iloc[:, :-1].values
label = data.iloc[:, [-1]].values

transformer = ColumnTransformer(transformers=[(
    'OneHot',
    OneHotEncoder(),
    [3])], remainder='passthrough')

features = transformer.fit_transform(features.tolist())
features = features.astype(float)

x_train, x_test, y_train, y_test = train_test_split(features, label, test_size=0.20, random_state=1)

print("******Decision Tree Regressor********")
for i in range(3, 8):
    dtr = DecisionTreeRegressor(max_depth=i)
    dtr.fit(x_train, y_train)
    print("max_depth= ", i)
    print("training score: ", dtr.score(x_train, y_train))
    print("testing score: ", dtr.score(x_test, y_test))

print()
print("*****Random Forest Regressor*********")
for i in range(3, 11):
    rfr = RandomForestRegressor(n_estimators=i)
    rfr.fit(x_train, y_train.ravel())
    print("n_estimator= ", i)
    print("Training score = ", rfr.score(x_train, y_train))
    print("Testing score = ", rfr.score(x_test, y_test))

lin_model = LinearRegression()
lin_model.fit(x_train, y_train)

y_train_predict = lin_model.predict(x_train)
rmse = (np.sqrt(mean_squared_error(y_train, y_train_predict)))
r2 = r2_score(y_train, y_train_predict)

print()
print("******Linear Regression************")
print("The model performance for training set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
print("\n")

y_test_predict = lin_model.predict(x_test)
rmse = (np.sqrt(mean_squared_error(y_test, y_test_predict)))
r2 = r2_score(y_test, y_test_predict)

print("The model performance for testing set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))

print()
print("Comparing the 3 models")
print("--------------------------------------")

dtr = DecisionTreeRegressor(max_depth=5)
dtr.fit(x_train, y_train)
dt_pred = dtr.predict(x_test)
dt = pd.DataFrame({'Real Profit Values': y_test.reshape(-1), 'Predicted Profit Values': dt_pred.reshape(-1)})
print("Decision Tree Regressor:")
dt["Difference"] = dt["Real Profit Values"] - dt["Predicted Profit Values"]
print(dt)
print("Average Difference: ", sum(dt["Difference"]) / len(dt["Difference"]))

rfr = RandomForestRegressor(n_estimators=5)
rfr.fit(x_train, y_train.ravel())
rfr_pred = rfr.predict(x_test)
rf = pd.DataFrame({'Real Profit Values': y_test.reshape(-1), 'Predicted Profit Values': rfr_pred.reshape(-1)})
print("Random Forest Regressor")
rf["Difference"] = rf["Real Profit Values"] - rf["Predicted Profit Values"]
print(rf)
print("Average Difference: ", sum(rf["Difference"]) / len(rf["Difference"]))

df = pd.DataFrame({'Real Profit Values': y_test.reshape(-1), 'Predicted Profit Values': y_test_predict.reshape(-1)})
print("Linear Regression Model")
df["Difference"] = df["Real Profit Values"] - df["Predicted Profit Values"]

print(df)
print("Average Difference: ", sum(df["Difference"]) / len(df["Difference"]))
