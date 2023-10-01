Day -16 Car price Prediction using Random Forest

import pandas as pd

#Load Dataset from local directory

from google.colab import files
uploaded = files.upload()

dataset = pd.read_csv("Car_prediction.csv")
dataset = dataset.drop(['car_ID'],axis=1)
print(dataset)

print(dataset.shape)
print(dataset.head(5))

#segregate

xdata = dataset.drop(['price'],axis = 'columns')
#x = dataset.drop('price',axis = 'columns')
numericalcols = xdata.select_dtypes(exclude=['object']).columns
x = xdata[numericalcols]
print(x)

y = dataset['price'].values
print(y)


#Scaling the Independent Variables

from sklearn.preprocessing import scale
cols = x.columns
X = pd.Dataframe(scale(x))
x.columns = cols
print(x)

#splitting train and test

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test  = train_test_split(x,y,test_size=0.25,random_state=0)

#Training using Random Forest

from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
model.fit(x_train,y_train)

#Evaluating The model

ypred = mdoel.predict(x_test)

from sklearn.metrics import r2_score
r2score = r2_score(y_test,ypred)
print("R2 Score " , r2score*100)




































































