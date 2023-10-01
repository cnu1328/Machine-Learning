import pandas as pd
import numpy as np

from google.colab import files
uploaded = files.upload()

dataset = pd.read_csv("DigitalAd_dataset.csv")

print(dataset.shape)
print(dataset.head(5))

#segregating

x = dataset.iloc[:,:-1].values
print(x)

y = dataset.iloc[:,-1].values
print(y)

#split train and test

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=0)

#Feature scaling

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

#Training

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(random_state=0)
model.fit(X_train,y_train)


#prediction
y_pred = model.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),1))

#Evaluating

from sklearn.metrics import confusion_matrix

cs = confusion_matrix(y_test,y_pred)
print("Confusion Matrix : ")
print(cs)

from sklearn.metrics import accuracy_score

print("Accuracy : {0}%".format(accuracy_score(y_test,y_pred)*100))


































