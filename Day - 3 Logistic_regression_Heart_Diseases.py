import warnings
warnings.filterwarnings('ignore')

#import basic libraries

import pandas as pd #To loab the dataset
import numpy as np #To do array operations

dataset = pd.read_csv("heart_Disease.csv")
print(dataset)

#Summerize the dataset
print(dataset.shape)
print(dataset.head(5))
print(dataset.tail(5))

#Segregating into x and y

x= dataset.iloc[:,:-1].values
print(x)

y = dataset.iloc[:,-1].values
print(y)


#Split to train and test

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=0)


#Feature Scaling

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)


#Training

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train,y_train)


#Prediction

y_pred = model.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))

#Confusion matrix and accuracy
from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(y_test,y_pred)
print("Confussion Matrix : ")
print(cm)

print("Accuracy of the model : {0}%".format(accuracy_score(y_test,y_pred)*100))


#Predicting the newPatient have heart diesase or not

age = int(input("Enter patient Age : "))
sex = int(input("Enter sex : 1 for male | 0 for female"))
cp = int(input("Enter CP : "))
bp = int(input("Enter BP : "))
chol = int(input("Enter Cholestrol : "))
fbs = int(input("Enter FBS : "))
restecg = int(input("Enter RESTECG : "))
thalach = int(input("Enter THALACH : "))
exang = int(input("Enter exang : "))
oldpeak = float(input("Enter oldpeak : "))
slope = int(input("Enter slope : "))
ca = int(input("Enter Ca : "))
tahl = int(input("Enter thal : "))
newPatient = [[age,sex,cp,bp,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,tahl]]
result = model.predict(sc.transform(newPatient))
print(result)
if result ==1:
    print("Patient have Heart Disease")
else:
    print("Patient don't Have Heart Disease")
    





