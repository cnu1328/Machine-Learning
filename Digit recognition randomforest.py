#import basic library

import pandas as pd
import numpy as np

from google.colab import drive
drive.mount('/content/gdrive')

filename = '/content/gdrive/My Drive/MachinelearningMasterClass/digit.csv'
dataset = pd.read_csv(filename)

print(dataset.shape)
print(dataset.head(5))

#segregate

x = dataset.iloc[:,1:]
print(x)
print(x.shape)

y = dataset.iloc[:,0]
print(y)

#split to train and test

# Training

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(x_train,y_train)


y_pred = model.predict(x_test)
#accuracy

import matplotlib.pyplot as plt
index = 10
print("predicted "+ str(model.predict(x_test)[index]))
plt.axis('off')
plt.imshow(x_test.iloc[index].values.reshape((28,28)),cmap='gray')
plt.show()
































