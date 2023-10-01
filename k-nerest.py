import pandas as pd
import numpy as np

from google.colab import files
uploaded = files.upload()


dataset = pd.read_csv("salary.sv")

print(dataset.shape)
print(dataset.head(5))

#mapping salary data to Binary value

income_set = set(dataset['income'])
dataset['income'] = dataset['income'].map({'<=50K':0,'>50K':1}).astype(int)
print(dataset.head(20))

#segregating to x and y

x = dataset.iloc[:,:-1].vlaues
print(x)

y = dataset.iloc[:,-1].values
print(y)


#splitting Dataset into Train & Test

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test= train_test_split(x,y,test_size=0.25, random_state=0)


#feature scaling

sc = StandardScaler()
X_train = sc.fit_tranform(x_train)
X_test = sc.transform(x_test)
print(X_train)


#finding the best k value

error = []
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

#Calculating error for k values between 1 and 40
for i in range(1,40):
    model = KNeighborsClassifier(n_neighbors=i)
    model.fit(X_train,y_train)
    pred_i = model.predict(X_test)
    error.append(np.mean(pred_i != y_test))

plt.figure(figsize=(12,6))
plt.plot(range(1,40),error,color='red',linestyle='dashed',marker='o',markerfacecolor='blue',markersize=10)

plt.title("Error RAte k value")
plt.xlabel("K value")
plt.ylabel("Mean Error")

#Training

from sklearn.neigbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors ='Take from graph'6,metric= 'minkowski',p=2)
model.fit(X_train,y_train)

#Predicting

age = int(input("Enter new Employee's Age : "))
adu = int(input("Enter new Employee's Education : "))
cg = int(input("Enter new Emplyee's Capital Gain : "))
wh = int(input("Enter New Emplyee's Hour's Per week : "))

newEmp = [[age,edu,cg,wh]]
result= model.predict(sc.transform(newEmp))
print(result)
if result==1:
    print("Employee migt got Salary above 50k")
else:
    print("Customer might not got Salary above 50k")


#Prediction for all Test Data

y_pred - model.predict(X_test)
print(np.concatenate((y_pred.reshae(len(y_red),1),y_test.reshape(len(y_test),1)),1))



#Evaluating Model-Confustion Matrix

from sklearn.metrics import confusion_matrix,accuracy_score
cm=confusion_matrix(y_test,y_pred)

print("Confusion Matrix")
print(cm)

print("Accuracy of the Model : {0}%".format(accuracy_score(y_test,y_pred)*100))





























