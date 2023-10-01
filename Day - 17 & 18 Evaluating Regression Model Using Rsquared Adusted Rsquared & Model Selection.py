#Day - 17 & 18 Evaluating Regression Model Using Rsquared Adusted Rsquared & Model Selection


#import libraries

import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

#Load dataset

from google.colab import files
uploaded = files.upload()


dataset = pd.read_csv("dataset.csv")

print(dataset.shape)
print(dataset.head(5))

#Visualize Dataset

plt.xlabel('area')
plt.ylabel('price')
plt.scatter(dataset.area,dataset.price,color='red',marker='*')
plt.show()

#segregate dataset x and y

x = dataset.drop('price',axis='columns')
print(x)

y = dataset.price.vaues

print(y)

#split dataset

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 20, random_state=0)

#Training and testing Data

model = LinearRegression()

model.fit(x_train,y_train)


#Visualinhg
plt.scatter(x,y,color='red',marker='*')
plt.plot(x,model.predict(x))
plt.title('Linear Regression')
plt.xlabel('Area')
plt.ylabel('price')
plt.show()



#Day - 18 Model Selection

importing libraries

import numpy as np
import pandas as pd
from matplotlib.pyplot as plt


dataset = pd.read_csv('dataset.csv')
x = datasett.iloc[:,:-1].values
y = dataset.iloc[:,-1].values
ysvm = y.reshape(len(y),1)#for svm we need to reshape

#Splitting the dataset into the |Training set and Test Set

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)
x_trainsvm,x_testsvm,y_trainsvm,y_testsvm = train_test_split(x,ysvm,test_size=0.2,random_state=0)


#importing Machine Learning Algorithms

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import Polynomialfeatures
from sklean.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

#Initializing Different Regression Algorithms

from sklearn.preprocessing import StandardScaler

modelLR = LinearRegression()

poly_reg = PolynomialFeatures(degree=4)
x_poly = ploy_reg.fit_transform(x_train)
modelPLR = LinearRegression()

modelRFR = RandomForestRegressor(n_estimators = 10, random_state = 0)
modelDTR = DecisionTree Regressor(random_state=0)

modelSVR = SVR(kernel = 'rbf')

sc_x = StandardScaler()
sc_y = StanndardScaler()
x_trainsvm = sc_x.fit_transform(x_trainsvm)
y_trainsvm = sc_y.fit_transform(y_trainsvm)



#Training REgression algorithm

modelLR.fit(x_train,y_train)
modelPLR.fit(x_poly,y_train)
modelRFR.fit(x_trian,y_train)
modelDTR.fit(x_train,y_train)
modelSVR.fit(x_trainsvm,y_trainsvm)

#Predicting the Test Set for Validation

modelLRy_pred = modelLR.predict(x_test)

modelPLRy_pred = modelPLR.predict(ploy_pred.transform(x_test))
modelRFRy_pred = modelRFR.predict(x_test)
modelDTRy_pred = modelDTR.predict(x_test)
modelSVRy_pred = sc_y.inverse_transform(modelSVR.predict(sc_x.transform(x_test)))


#Evaluating The Model Performance

from sklearn.metrics import r2_score
print('Linear Regression Accuracy : {}'.format(r2_score(y_test,modelLRy_pred)))
print('Polynomial Regression Accuracy : {}'.format(r2_score(y_test,modelPLRy_pred)))
print('Random Forest Regression Accuracy : {}'.format(r2_score(y_test,modelRFRy_pred)))
print('Decision Tree Regression Accuracy : {}'.format(r2_score(y_test,modelDTRy_pred)))
print('Support Vector Regression Accuracy : {}'.format(r2_score(y_test,modelSVRy_pred)))

















































































































