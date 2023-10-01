#Day-11 House price prediction using Linear REgresion - singleVariable

#Import laibraires
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

#Load dataset
from google.colab import files
uploaded = files.upload()

#Load Dataset

dataset = pd.read_csv('dataset.csv')

print(dataset.shape)
print(dataset.head(5))

#visualize the dataset

plt.xlabel('Area')
plt.ylabel('Price')
plt.scatter(dataset.area,dataset.price,color='red',marker='*')

#segregate

x = dataset.drop('price',axis='columns')

y = dataset.price

#Training

model = LinearRegression()
model.fit(x,y)

#prediction

x = 4685
LandAreainSqFt=[[x]]
predictedmodelresult = model.predict(LandAreainSqFt)
print(predictedmodelresult)

m= model.coef_
print(m)

b = model.intercept_
print(b)

y = m*x+b

print("The price of {0} Square feet Land is : {1}".format(x,y[0]))



































