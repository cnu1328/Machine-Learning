import warnings
warnings.filterwarnings("ignore")

import pandas as pd #to read the datasheet
import numpy as np#To perform array

dataset = pd.read_csv("salary.csv")
print(dataset)

#Summerize the dataset

print(dataset.shape)
print(dataset.head(5))
print(dataset.tail(5))


#Mapping the salary data to binary
income_set = set(dataset['Income'])
dataset["Income"] = dataset["Income"].map({'<=50K':0,'>50K':1}).astype(int)
print(dataset.head(10))


#segregating to x and y

x = dataset.iloc[:,:-1].values
print(x)
y=dataset.iloc[:,-1].values
print(y)

#Splitting Dataset into Train&Test

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=0)


#Feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)
print(X_train)

#finding the best k value

error = []
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

#calculating error for values between 1 and 40
for i in range(40,90):
    model = KNeighborsClassifier(n_neighbors=i)
    model.fit(X_train,y_train)
    pred_i = model.predict(X_test)
    error.append(np.mean(pred_i!=y_test))

plt.figure(figsize=(12,6))
plt.plot(range(40,90),error,color='red',linestyle='dashed',marker='o',markerfacecolor='blue',markersize=10)

plt.title("Error Rate K Value")
plt.xlabel("K value")
plt.ylabel("Mean Error")

plt.show()


#Training the data

from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=42,metric='minkowski',p=2)
model.fit(X_train,y_train)

#Prediction of all Text Data

y_pred = model.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))


#Evaluating Model-Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test,y_pred)
print("Confusion Matrix : ")
print(cm)

print("Accuracy of the Model : {0}%".format(accuracy_score(y_test,y_pred)*100))

#Predicting new employ get lessthan or equal to 50k or grater than 50k

age = int(input("Enter New Employee's Age : "))
adu = int(input("Enter New Employee's Education : "))
cd = int(input("Enter New Employee's Capital Gain : "))
wh = int(input("Enter New Employee's Hour's Per week : "))

newEmp = [[age,adu,cd,wh]]
result = model.predict(sc.transform(newEmp))

print(result)

if result ==1:
    print("Employee Might Got Salary Above 50K")
else:
    print("Employee Might Not Got Salary Above 50K")


