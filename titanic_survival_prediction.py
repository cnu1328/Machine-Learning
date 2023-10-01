#Day-6 Titanic Survival Prediction Using NAIVE BAYES

#importing libraries
import pandas as pd
import numpy as np

from google.colab import files
uploaded = files.upload()

#Load dataset
dataset = pd.read_csv('titanicsurvival.csv')
print(dataset)
print(dataset.shape)
print(dataset.head(5))

#Mapping Text Data to Binary value
sex_set = set(dataset['Sex'])
dataset['Sex']=dataset['Sex'].map({'female':0,'male':1}).astype(int)
print(dataset.head(5))

#Segregate Dataset

x = dataset.drop('Survied'axis='columns')
print(x)
y = dataset.Survived
print(y)

#identify and replace not availabe values
x.columns[x.isna().any()]
x.Age=x.Age.fillna(x.Age.mean())

#Test

x.columns[x.isna().any()]

#Spliting train and test

from sklearn.model_selection import train_test_split
x_train,x+test,y_train,y_test =

#Training

from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(X_train,y_train)

#Evaluated

y_pred = model.predict(x_test)
print(np.column_stack((y_pred,y_test())))

#Accuracy of our Model

from sklearn.metrics import accuracy_score
print("Accuracy of the Model: {0}%".format(accuracy_score(y_test,y_pred)*100))


#Predicting, Wheather Person Survived or Not

pclassNo = int(input("Enter person's pclass number : "))
gender = int(input("Enter person's gender 0-female,1-male(0 or1) : "))
age = int(input("Enter person's Age : "))
fare = float(input("Enter Person's Fare " ))
person = [[pclassNo,gender,age,fare]]
result = model.predict(person)
print(result)

if result ==1:
    print("Person might be survived")

else:
    print("Person mgiht not be survived")
    




































