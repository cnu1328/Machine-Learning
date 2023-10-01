#Day-12 Salary prediction using polynomial Regression

import pandas as pd

from google.colab import files

uploaded = files.upload()

#Load Dataset

dataset = pd.read_csv("dataset.csv")

print(dataset.shape)
print(dataset.head(5))

#segregate the dataset
x = dataset.iloc[:,:-1].values
print(x)
y = dataset.iloc[:,-1].values

print(y)


#Training dataset using Linear |Regression
from sklearn.linear_model import LinearRegression
modelLR = LinearRegression()
modelLR.fit(x,y)


#Visulainzing Lienar Regression Rsults

import matplotlib.pyplot as plt
plt.scatter(x,y,color='red')
plt.plot(x,modelLR.predict(x))
plt.title("Linear Regression")
plt.xlabel("Level")
plt.ylabel("Salary")
plt.show()

#convert x to polynomail format(x^n)

from sklearn.preproce3ssing import PolynomialFeatures
modlPR = PolynomialFeatures(degree=4)
xPloy = modelPR.fit_transform(x)

#Train Same Linear Regression with x-polynomial instead of x
modelPLR = LinearRegression()
modelPLR.fit(xPoly,y)


#Visualizing polynomial resgression

plt.scatter(x,y,color='red')
plt.plot(x,modelPLR.predit(modelPR.fit_transform(x)))
plt.title("Ploynomila Regression")
plt.xlabel('Level')
plt.ylabel('Salary')
plt.show()

#Prediction using Polynomial Regression

x = 5 salaryPred = modelPLR.predict(modelPR.fit_transform([[x]]))
print("Salary of a person with level {0} is {1}".format(x,salaryPred))


























