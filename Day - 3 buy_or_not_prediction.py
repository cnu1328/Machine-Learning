import warnings
warnings.filterwarnings('ignore')


import pandas as pd#used for loading the dataset
import numpy as np#to perform array

dataset = pd.read_csv("DigitalAd_dataset.csv")
print(dataset)

#summerize the dataset

print(dataset.shape)
print(dataset.head(5))
print(dataset.tail(5))

#devide the dataset into input and output
x=dataset.iloc[:,:-1].values
print(x)

y=dataset.iloc[:,-1].values
print(y)

#To split train and test
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)
print(x_train)
print(x_test)
print(y_train)
print(y_test)

#Scaling according to feature or standard

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train = sc.fit_transform(x_train)
print(X_train)
X_test=sc.fit_transform(x_test)
print(X_test)

#Training the Dataset

from sklearn.linear_model import LogisticRegression
model=LogisticRegression()
model.fit(X_train,y_train)

y_pred= model.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))


#confusion matrices

from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(y_test,y_pred)
print("Confusion Matrix : ")
print(cm)
print("Accuracy of the Dataset : {0}%".format(accuracy_score(y_test,y_pred)*100))




#predicting, Whether new customer with Age & Salary will Buy or Not
age= int(input("Enter new Customer's Age : "))
sal =int(input("Enter new Customer's Salary : "))
newCust = [[age,sal]]
result = model.predict(sc.transform(newCust))
print(result)
if result==1:
    print("Customer will Buy")

else:
    print("Customer won't Buy")
































      
