#Day-6 Titanic Survival Prediction Using NAIVE BAYES

import pandas as pd
import numpy as np

dataset = pd.read_csv("TitanicSurvival.csv")
print(dataset)

#Summerize the data

print(dataset.shape)
print(dataset.head(5))

#Mapping Text data to Binary Value

dataset['survived']=dataset['survived'].map({'yes':1,'no':0}).astype(int)
dataset['sex'] = dataset['sex'].map({'female':0,'male':1}).astype(int)
dataset['passengerClass'] = dataset['passengerClass'].map({'1st':1,'2nd':2,'3rd':3}).astype(int)
print(dataset)

#segregate the datainto x and y

x = dataset.iloc[:,2:6]
print(x)

y = dataset.survived.values
print(y)


#Identify and Replace Not availabe Values
print(x.columns[x.isna().any()])
x.age = x.age.fillna(x.age.mean())

print(x.columns[x.isna().any()])

X = x.values
print(X)


#spliting train and test

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size = 0.25,random_state=0)



#Training

from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(x_train,y_train)

#Evaluted
y_pred = model.predict(x_test)
print(np.column_stack((y_pred,y_test)))

#accuracy of the model

from sklearn.metrics import accuracy_score,confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print("Confusion Matrix Of The Model : ")
print(cm)

print("Accuracy Of The Model : {0}%".format(accuracy_score(y_test,y_pred)*100))


#Predicting Whether the person survived or not

gender = int(input("Enter person's gender 0-female,1-male(0 or 1) : "))
age = int(input("Enter person's Age : "))
pc = int(input("Enter person's pclass number : "))

newPerson = [[gender,age,pc]]
result = model.predict(newPerson)
print(result)

if result==1:
    print("Person might be survived")
else:
    print("Person might not be survived")







































































