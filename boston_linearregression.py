import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt


%matplotlib inline

boston_dataset = load_boston()

#describes the dataset
boston_dataset.DESCR

print(boston_dataset.keys())
#data is the information, target = prices of houses - dependent variable
#feature_names is th real name of the columns

boston = pd.DataFrame(boston_dataset.data, columns = boston_dataset.feature_names)
boston.head()
#shows top 5

boston.isnull().sum()
#checks the null values

#check the distribution
sns.set(rc={'figure.figsize':})

#heatmap

plt.figure(figsize=(20, 5))

features == ['LSTAT', 'RM']
target = boston['MEDV']

for i, col in enumerate(features):
    plt.subplot(1, len(features), i+1)
    x=boston[col]
    y = target
    plt.scatter(x, y, marker='o')
    plt.title(col)
    plt.xlabel(col)
    plt.ylabel('MEDV')