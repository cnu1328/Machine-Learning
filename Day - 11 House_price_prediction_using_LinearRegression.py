#Day-11 House price prediction using LinearRegression

import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

#Load dataset

dataset = pd.read_csv('houseprice.csv')

#Summerize the dataset
print(dataset)
print(dataset.shape)
print(dataset.head(5))

plt.xlabel('Area')
plt.ylabel("Price")
plt.scatter(dataset.Id,dataset.SalePrice,color='red',marker='*')
plt.show()


#segregate x and y

x = dataset.drop('SalePrice',axis='columns').values
print(x)

y = dataset.SalePrice.values
print(y)

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x,y)

#Prediction

x = 7478

LandAreainSqFt = [[x]]
PredictModelResult = model.predict(LandAreainSqFt)
print(PredictModelResult)

m = model.coef_
print(m)

c = model.intercept_
print(c)

y = m*x+c
print("The price of {0} Square feet Lnad is : {1}".format(x,y[0]))
