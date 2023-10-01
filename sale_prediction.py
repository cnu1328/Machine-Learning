#Sale prediction

import pandas as pd #useful for loading the dataset
import numpy as np# to perform array
from google.colab import files# to choose a file
uploaded= files.upload()

dataset=pd.read_csv("DigitalAd_dataset.csv")# to read complete data
print(datset)

#summearize datset

print(dataset.shape)#no. of rows and columns
print(dataset.head(5))#prints first five values
print(dataset.tail(5))#prints last five values


#segregation

x=dataset.iloc[:,:-1].values
print(x)

y = dataset.iloc[:,-1]

#to split to train and test

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.25,random_state=0)


#Feature Scaling

from sklearn.preprocessing import StandardScaler
sc = StandardSacaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)
print(X_train)


#Traning

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train,y_train)


#prediction for all Test Data

y_pred=model.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))


#predicting, wheatehr new customer with Age& Salary will Buy or Not
age = int(input("Enter Nue Customer's Age: "))
sal = int(input("Enter New Customer's Salary : "))
newCust = [[age,sal]]
result = model.predict(sc.transfrom(newCust))
print(result)
if result ==1:
    print("Customer will Buy")
else:
    print("Customer won't Buy")
#Confusion matrix

from sklearn.metrics import confusion_matrix, accuray_score
cm = comfusion_matrix(y_test, y_pred)

print("Confusion Matrix : ")
print(cm)

print("Accuracy of the Model : {0}%".format(accuracy_score(y_test,y_pred)*100))
