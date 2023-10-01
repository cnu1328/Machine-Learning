#Day-21 Plant Iris clustering using Principal Component Analysis

from sklearn import datasets

import matplotlib.pyplot as plt

dataset = datasets.load_iris()

x = dataset.data
print(x)
y = dataset.target
print(y)
names = dataset.target_names
print(names)

#Principal component Analysis found in sklearn.decomposition library

from sklearn.decomposition import PCA

model = PCA(n_components=2) #Number of components to keep

y_means = model.fit(x).transform(x)
print(y_means)

#Variance Percentage

plt.figure()

colors = ['red','green','orange']

for color,i,target_name in zip(colors,[0,1,2],names):
    plt.scatter(y_means[y==i,0],y_means[y==i,1],color = color, lw =2, label = target_name)

plt.title('IRIS Clustering')
plt.show()
