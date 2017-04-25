# CS205 2017 Spring Final Project 
Group Members (in alphabetical order): <br /> 
Jiahua Guo  <br />
Jiachen Song  <br />
Xinyuan Wang  <br />
Jiawei Zhuang  <br />

# Table of Contents
  * [Introduction](#introduction)
  * [Paralleled Kmeans Algorithms](#paralleled-kmeans-algorithms)
  * [Applications](#applications)
  * [Advanced Features](#advanced-features)
  
## Introduction

## Paralleled Kmeans Algorithms

## Applications

## Advanced Features
  
  
  
### Parallel software solution
We choose MPI + OpenMP/OpenACC/CUDA as our heterogenous computing environment. 

### Data science problem
Many huge data sets are now publicly available. There are several ways to turn those large amounts of data into useful knowledge. 
Here we focus on exploratory data analysis, or unsupervised machine learning, which means finding structural information without prior knowledge.

Among all the unsupervised learning methods, k-means is a commonly used algorithm, which partitions observations into k clusters in which each 
observation belongs to the cluster with the nearest mean. Finding the minimum of a k-means cost function is a NP-hard problem when the dimension 
d>1 and the number of clusters k>1. Scientists came up with several heuristic methods to find the local minimum, but the process is still highly 
computationally-intensive, especially with huge data sets. We want to implement a parallel version of a k-means heuristic method on a cluster of machines, 
to significantly speed up the computing time of the clustering process, without any reduction on the accuracy rate of the clustering model.

### Application of kmeans clustering: Detecting abnormal meteorology events
In this part, we would like to use k-means cluster technique to examine a type of climate events, called sudden stratospheric warmings (SSWs). The climatological zonal winds in the stratosphere are generally westerly and their strength increases with height. These winds form the \"polar night jet\" vortex, and can be very persistent during winters, as shown in fig(a). However, at times this zonal-mean configuration is dramatically disturbed, as shown in fig(b) and fig(c), with latitudinal temperature gradient and zonal-mean winds at the pole being reversed.

In the past, these pheonomena have been arbitrarily defined using a variety of different criteria involving winds, temperatures, and measures of the vortex shape. Using thresholds can be a powerful and useful way to understand variability, but more or less a subjective way in terms of choosing the thresholds. k-means clustering is a method of identifying different states in a completely objective manner with no preconceived notion of the groups and no preselection on the basis of known influencing factors.

k-means clustering technique is more useful than hierarchical clustering for this type of problems, because k-means clustering easily allows for uneven groups, whereas hierachical clusetering tends to determine groups of similar sizes. 

In addition, this type of problems usually involves a very large dataset with very high dimensions, e.g. more than 17,000 data points with 252 dimensions in this example, therefore a simple clustering technique such as k-means is very useful.
![Silhouette Value](Data_Analysis/figures/svalue.png)
![Temperature Anomaly](Data_Analysis/figures/T.png)
![Vortex Structure](Data_Analysis/figures/PV.png)
### Some data set options 

Hubway system data: <br />
https://www.thehubway.com/system-data

Airbnb data: <br />
http://data.beta.nyc/dataset/inside-airbnb-data

