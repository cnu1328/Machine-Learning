#Handwritten Digit REcognition

#importing libraries
import numpy as np
from sklearn.datasets import load_digits

dataset = load_digits()
print(dataset.data)

print(dataset.data.shape)
print(dataset.images.shape)
print(dataset.target)

dataimageLength = len(dataset.images)
print(dataimageLength)

#segregating x and y value

x = dataset.images.reshape((dataimageLength,-1))
print(x)
y = dataset.target

#splitting the dataset into Train and Test

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.25,random_state=0)

#Trainiing
from sklearn import svm

model = svm.SVC()
model.fit(x_train,y_train)


#Prediction of test Data
y_pred = model.predict(x_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))

#Accuracy
from sklearn.metrics import accuracy_score
print("Accuracy of the model : {0}%".format(accuracy_score(y_test,y_pred)*100))


#Predicting, What the digit from test data

n = int(input("Enter n value : "))
result = model.predict(dataset.images[n].reshape((1,-1)))
print(result)

import matplotlib.pyplot as plt
plt.gray()
plt.matshow(dataset.images[n])
plt.show()


#Playing with different methods

from sklearn import svm
#Create models

model1 = svm.SVC(kernel="linear")
model2=svm.SVC(kernel='rbf')
model3=svm.SVC(gamma=0.001)
model4 = svm.SVC(gamma=0.001,coef0=0.1)

#fit the model

model1.fit(x_train,y_train)
model2.fit(x_train,y_train)
model3.fit(x_train,y_train)
model4.fit(x_train,y_train)

#Prediting the model

y_predModel1 = model1.predict(x_test)
y_predModel2 = model2.predict(x_test)
y_predModel3 = model3.predict(x_test)
y_predModel4 = model4.predict(x_test)

#Printing accuracy

print("Accuracy of the model : {0}%".format(accuracy_score(y_test,y_predModel1)*100))
print("Accuracy of the model : {0}%".format(accuracy_score(y_test,y_predModel2)*100))
print("Accuracy of the model : {0}%".format(accuracy_score(y_test,y_predModel3)*100))
print("Accuracy of the model : {0}%".format(accuracy_score(y_test,y_predModel4)*100))































