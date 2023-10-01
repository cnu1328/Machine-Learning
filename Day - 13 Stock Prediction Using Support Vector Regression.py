
Day - 13 Stock prediction Using Support Vactor Regression

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset

dataset = pd.read_csv("data.csv")
x = dataset.iloc[:,:-1].values
y = datset.iloc[:,-1].values

print(y)

y = y.reshape(len(y),1)
print(y)

#splitting The dataset into the Trainig set and test set

from sklearn.model_selection import train_test_split

#Feature Scalling
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
x_train = sc_x.fit_transform(x_train)
y_train = sc_y.fit_transform(y_train)

#training the SVR model on the Training set

from sklearn.svm import SVR
regressor = SVR(kernerl = 'rbf')
regressor.fit(x_train,y_train)


#Prediction
y_pred = sc_y.inverse_transform(regressor.predict(sc_x.transfor(x_test)))
mp.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1))))


#Evaluation the Model Performance

from sklearn.metrics import r2_score
r2_score(y_test,y_pred)




































