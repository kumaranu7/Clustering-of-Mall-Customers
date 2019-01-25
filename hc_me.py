#importing the libraries
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
#importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:, [3, 4]].values

#having a look over the dataset we see that we don't need feature sacling, or data preprocessing as for now.

#using dendograms to find optimal no.of clusters
import scipy.cluster.hierarchy as sch
dendogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
#visualing the dendogram for optimal number of clusters
plt.title('Dendogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean Dist.')
plt.show()
#fitting hierarchical clustering to the mall dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
#predicting the results on test test
y_hc = hc.fit_predict(X) 

#visualising th clusters in 2-Dimensions
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s=100, c = 'red', label = 'Careful')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s=100, c = 'blue', label = 'Standard')
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s=100, c = 'green', label = 'Target')
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s=100, c = 'cyan', label = 'Careless')
plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s=100, c = 'magenta', label = 'Sensible')
plt.scatter(hc.cluster_centers_[:,0], hc.cluster_centers_[:,1], s=300, c = 'yellow', label = 'centroid')
plt.title('Cluster of clients')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.legend()
plt.show()







