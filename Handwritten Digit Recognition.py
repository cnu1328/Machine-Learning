#Day-5

#Importing Basic Libraries

import numpy as np
from sklearn.datasets import load_digits

#Load Dataset
dataset= load_digits()

#summerize Dataset
print(dataset.data)
print(dataset.target)

print(dataset.data.shape)
print(dataset.images.shape)

dataimageLength = len(dataset.images)
print(dataimageLength)


#Visualize the Dataset
n=85 # No. of sample out of Samples total 1797

import matplotlib.pyplot as plt

plt.gray()
plt.matshow(dataset.images[n])
plt.show()

dataset.images[n]


#Segregate Dataset into X(Input/IndependentVariable)& Y(Output/DependentVariable)
#Input-Pixel|Output-class

x= dataset.images.reshape((dataimageLength,-1))
print(x)
y = dataset.target
print(y)


Splitting Dataset Into Train & Test

from sklearn.model_selection import train_test_split

X_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.25,random_state=0)
print(X_train.shape)
print(X_test.shape)

#Training

from sklearn import svm
model = svm.SVC()

modl.fit(x_train,y_train)


#Predicting, what the digit is from Test Data
n=1000
result = model.predict(dataset.images[n],reshape((1,-1)))
plt.imshow(dataset.images[n],cmap=plt.cm.gray_r,interpolation='nearest')
print(result)
print('\n')
plt.axis('off')
plt.title('%i' %result)
plt.show()



#Prediction for TestData
y_pred= model.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))

#Evaluate Model-Accuracy Score
from sklearn.metrics import accuracy_score
print('Accuracy of the model : {0}%'.format(accuracy_score(y_test,t_pred)*100))


#Play with the Different Method
from sklearn import svm
model1=svm.SVC(kernel='linear')
model2=svm.SVC(kernel='rbf')
model3=svm.SVC(gamma=0.001)
model4=svm.SVC(gamma=0.001,c=0.1)

model1.fit(X_train,y_train)
model2.fit(X_train,y_train)
model3.fit(X_train,y_train)
mdoel4.fit(X_train,y_train)

y_preModel1 = model1.predict(X_test)
y_preModel2= model.predict(X_test)
y_predModel3= model.predict(X_test)
y_predModel4 = model4.predict(X_test)

print("Accuracy of the Model 1 : {0}%".format(accuracy_score(y_test,y_preModel1)*100))

print("Accuracy of the Model 2 : {0}%".format(accuracy_score(y_test,y_preModel2)*100))

print("Accuracy of the Model 3 : {0}%".format(accuracy_score(y_test,y_preModel3)*100))

print("Accuracy of the Model 4 : {0}%".format(accuracy_score(y_test,y_preModel4)*100))


