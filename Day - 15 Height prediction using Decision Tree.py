import pandas as pd
import numpy as np

import matplotlib.pylot as plt

from google.colab import files

uploaded = files.upload()

dataset = pd.read_csv("dataset.csv")

print(dataset.shape)
print(datset.tail(5))

#Segregating Dataset into x and y

x = dataset.iloc[:,:-1].values

y = datset.iloc[:,-1].vlaues


Splitting

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20,random_state = 0)


#Training Dataset using Decision Tree

from sklearn.tree import DecisionTreeRegressor

model = DecisionTreeRegressor()
model.fit(x_train,y_train)


#Visualizng Graph

x_val = np.arrange(min(x_train),max(x_train),0.01)
x_val = x_val.reshape(len(x_val),1)
plt.scatter(x_train,y_train,color ='green')
plt.plot(x_val,model.predict(x_val),color = 'red')
plt.tilte('Height prediction uisng Decision Tree')
plt.xlabel('Age')
plt.ylable('Height')
plt.figure()
plt.show()


#Prediction for all test data for validation

y_pred = model.predict(x_test)

from sklearn.metrics import r2_score,mean_squared_error
mse = mean_squared_error(y-test,y_pred)
rmse = np.sqrt(mse)
print("Root Mean Square Error : ",rmse)
r2score = r2score(y_test,y_pred)
print('R2Score',r2score*100)



