import warnings
warnings.filterwarnings('ignore')

from sklearn.datasets import load_iris
import pandas as pd
import numpy as np

dataset = load_iris()
print(dataset.data)
print(dataset.data.shape)
print(dataset.target)

#Segregating the data into x and y

x = pd.DataFrame(dataset.data,columns=dataset.feature_names)
print(x)
y = dataset.target

#spliting to trian and test

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=0)

#finding best ma_depth value

accuracy = []

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

for i in range(1,10):
    model = DecisionTreeClassifier(max_depth=i,random_state=0)
    model.fit(x_train,y_train)
    pred = model.predict(x_test)
    score = accuracy_score(y_test,pred)
    accuracy.append(score)

plt.figure(figsize=(12,6))
plt.plot(range(1,10),accuracy,color='red',linestyle = 'dashed',marker = 'o',markerfacecolor='blue')
plt.title('Finding Best Max_Depth')
plt.xlabel('pred')
plt.ylabel('score')
plt.show()

#Training

from sklearn.tree import DecisionTreeClassifier

model =  DecisionTreeClassifierc

