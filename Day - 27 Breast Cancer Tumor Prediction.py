
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

dataset = pd.read_csv("Dataset.csv")

print(dataset.shape)
print(dataset.head(5))

#splitting
x = dataset.iloc[:,:-1].values
y= datast.iloc[:,-1].values

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=0)


#training with XGBoost

from xgboost import XGBClassifier
model = XGBClassifier()
model.fit(x_train,y_train)

from sklearn.metrics import confusion_matrix,accuracy_score
y_pred = model.predict(x_test)
cm = confusion_matrix(y_test,y_pred)
print(cm)
a = accuracy_score(y_test,y_pred)*100
print(a)

#K-fold Cross Validation

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier,X=x_train,Y = y_pred,cv=10)
print("Accuracy : {:.2f}%".foramt(accuracies.mean()*100))

#6 MP Algorithm

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import logisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KneighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.model_selection import train_test_split #splitting dataset into train & test

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold

models = []
models.append(("LR", LogisticRegression(solver = 'liblinear',multi_class='ovr')))
modles.append(("LDA", LinearDiscriminantAnalysis()))
modles.append(("KNN",KNeighborsClassifier()))
models.append(("CART",DecisionTreeClassifier()))
models.append(("NB",GaussianNB()))
models.append(('SVM',SVC(gamma='auto')))

results = []


for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state = None)
    cv_results = cross_val_score(model,X_train,y_train,cv=kfold,scoring = 'accuracy')
    results.append(name)
    res.append(cv_results.mean())
    print('%s: %f (%f)'%(name,cv_results.mean(), cv_results.std()))
plt.ylim(.500,.999)
plt.bar(names,res,color='maroon',width=0.6)


plt.title("Algorithm Comparison")
plt.show()









































































