import numpy as np
import pandas as pd
import math
import random

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn import mixture

X_normed=pd.read_csv("data/observation_sample.csv")
Y=pd.read_csv("data/label_sample.csv")

#use silhouette_values to choose best K
sil=[]
start = 5
end = 7
for n_clusters in range(start,end):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X_normed)
    cluster_labels=kmeans.labels_
    silhouette_values = silhouette_samples(X_normed,cluster_labels)

    silhouette_avg = silhouette_score(X_normed, cluster_labels)
    print("For n_clusters =", n_clusters,
              "The average silhouette_score is :", silhouette_avg)
    sil.append(silhouette_avg)



K = start+np.array(sil).argmax()

#Now we've found the K, do the clustering again
kmeans = KMeans(n_clusters=K, random_state=0).fit(X_normed)
cluster_labels=kmeans.labels_
silhouette_values = silhouette_samples(X_normed,cluster_labels)

silhouette_avg = silhouette_score(X_normed, cluster_labels)
print("For n_clusters =", K,
          "The average silhouette_score is :", silhouette_avg)


#find the purest(has the highest percentage for a cluster) cluster for the 7 labels, use their center as our final cluster center so that future data points could
#be classified

#build a table so that all the percentage could be recorded
#cluster Number  label_1  label_2   ...   label_7   
#0
#1
#...
#K-1 

table = pd.DataFrame(np.zeros((K,8)),columns=['label_1', 'label_2', 'label_3', 'label_4','label_5','label_6','label_7','size'])


pred = pd.DataFrame(data = kmeans.labels_, columns=['cluster'])
for k in range(K):
    for i in range(1,8):  
        c = Y.iloc[pred[pred['cluster']==k].index]
        table.loc[k,'label_'+str(i)] = c[c==i].shape[0]


# print(table)


target = open("data/output.txt", 'w')
target.write(str(K))
target.write("\n")
target.write(str(sil))
target.write("\n")



pred.to_csv("data/pred_labels.csv")
#select 7 clusters, set their centers for use
#1: 7
#2: 4
#3: 22
#4: 23
#5: 0
#6: 3
#7: 24
# selected_cluster  = [7,4,22,23,0,3,24]
# centers = kmeans.cluster_centers_[selected_cluster]



#test our classification accuracy 































