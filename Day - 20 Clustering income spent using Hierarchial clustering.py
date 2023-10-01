#Day-20 Clustering income spent using Hierarchial clustering

import pandas as pd

import matplotlib.pyplot as plt

#load dataset

dataset = pd.read_csv("Dataset.csv")

#summerize thedataset

print(dataset.shape)
print(dataset.describe())
print(dataset.head(5))


#Label Encoding

from sklearn import preprocessing

label_encoder = preprocessing.LabelEncoder()

dataset['Gender'] = label_encoder.fit_transform(dataset['Gender'])
dataset.head()


#Dendrogram Data Visualization

import scipy.cluster.hierarchy as clus

plt.figure(1,figsize = (16,8))
dendrogram = clus.dendrogram(clus.linkage(dataset,method ='ward'))

plt.title('Dendrogram Tree Graph')
plt.xlabel('Customers')
plt.ylabel('distances')
plt.show()

#fitting the Hierarchial Clustering to the dataset with n=5




































