import pandas as pd #to load dataset
import numpy as np # to array operations
from matplotlib import pyplot#To visualization

from google.colab import files
uploaded = files.upload()

dataset = pd.read_csv("data.csv")


print(dataset)

print(dataset.shape)
print(dataset.head(5))

dataset['diagnosis'] = dataset['diagnosis'].map({'B':0,'M':1}).astype(int)
print(dataset.head(5))


x = dataset.iloc[:,2:32].values
print(x)

y = dataset.iloc[:,1].values
print(y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test


#Feature Scaling

X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)


#Validating Some ML algorithm by its accuracy - Model Score
from sklearn.discriminant _analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedkFold


models = []
models.append(('LR',LogisticRegression(solver='livlinear',multi_class = 'ovr')))

models.append(('LDA',LinearDiscriminant()))

models.append(('KNN',KNeighborsClassifier()))

models.append(('CART',DesicisonTreeClassifier()))

models.append(('NB',GaussianNB()))

models.append(('SVM',SVC(gamma='auto')))


print(models)


results = []
names = []
res = []

for name,model in models:
    kfold = StratifiedKFold(n_splits=10,random_state = None)
    cv_results = cross_val_score(model,x_train,y_train,cv = kfold,scoring = 'accuracy')
    results.append(cv_results)
    names.append(name)
    res.append(cv_results.mean())
    print("%s : %f"%(name,cv_results.mean()))

pyplot.ylim(.900,.999)
pyplot.bar(names,res,color='maroon',width=0.6)
pyplot.title("Algorithm Comparision")
pyplot.show()


#Training & Prediction using the algorithm with high accuracy

model = LogisticRegression(solver = 'liblinear',multi_class='ovr')
model.fit(x_train,y_train)
value = [[]]
y_pred = model.predict(value)\print(y_pred)
    




































