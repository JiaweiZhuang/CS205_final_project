# Table of Contents
  * [Introduction](#introduction)
  * [Parallel Kmeans Algorithms](#parallel-kmeans-algorithms)
  * [Applications](#applications)
  * [Advanced Features](#advanced-features)
  
# Introduction
Many huge data sets are now publicly available. There are several ways to turn those large amounts of data into useful knowledge. 
Here we focus on exploratory data analysis, or unsupervised machine learning, which means finding structural information without prior knowledge.

Among all the unsupervised learning methods, k-means is a commonly used algorithm, which partitions observations into k clusters in which each 
observation belongs to the cluster with the nearest mean. Finding the minimum of a k-means cost function is a NP-hard problem when the dimension 
d>1 and the number of clusters k>1. Scientists came up with several heuristic methods to find the local minimum, but the process is still highly 
computationally-intensive, especially with huge data sets. We want to implement a parallel version of a k-means heuristic method on a cluster of machines, 
to significantly speed up the computing time of the clustering process, without any reduction on the accuracy rate of the clustering model.

# Parallel Kmeans Algorithms

## OpenMP, MPI and hybrid MPI-OpenMP parallelization

A typical approach for k-mean clustering is Expectation–Maximization (E–M). E-step assigns points to the nearest cluster center, while M-step set the cluster centers to the mean. 

### OpenMP

With OpenMP parallelization, only E-step can be directly parallelized. If M-step is directly parallelized with OpenMP pragmas, different data points might be added to one cluster at the same time, leading to Write-After-Write (WAW) harzard. Although it is possible to make drastic modifications to parallelize the M-step, it contradicts the basic idea of OpenMP that the serial code shoud be almost untouched. Therefore, we only focus on the E-step. 
[(View our OpenMP code)](Parallel_Algorithm/OpenMP/Kmean_omp.c)

Unsurprisingly, while the E-step scales well, the M-step even gets slower because of thread overheads. Although the M-step is not time-consuming in the serial case, it finally becomes the bottleneck when the number of cores gets large:
<p align="center">
<img src="Timing_Results/plots/OpenMP_scaling.jpg" width="720">
</p>

[(View the raw timing log)](Timing_Results/log/Blobs_OpenMP.log)

### MPI

With MPI, we can distribute data points to different processes using MPI_Bcast, and use MPI_Allreduce to exchange information whenever needed. Thus, both the E-step and the M-step can be parallelized. [(View our MPI code)](Parallel_Algorithm/MPI/Kmean_mpi.c)

This time, we get speed-up in both steps, so the overall scaling is better than OpenMP.
<p align="center">
<img src="Timing_Results/plots/MPI_scaling.jpg" width="720">
</p>

[(View the raw timing log)](Timing_Results/log/Blobs_MPI.log)

### Hybrid MPI-OpenMP

We simply add OpenMP pragmas to the MPI code, to get the hybrid version. This time we have many combinations of OpenMP threads and MPI processes to test. In general, we find that the speed-up depends on the product of the number of OpenMP threads (n_omp hereinafter) and the number of MPI processes (N_MPI hereinafter):

<p align="center">
<img src="Timing_Results/plots/hybrid_scaling.jpg" width="480">
</p>

[(View the raw timing log)](Timing_Results/log/Blobs_hybrid.log)

Interestingly, for N_MPI*n_omp=32, we have tested 4 cases (N_MPI,n_omp) = (32,1), (16,2), (8,4) or (4,8), and all of them have almost the same speed. 
[(see the exact time use in the last cell)](https://github.com/JiaweiZhuang/CS205_final_project/blob/master/Timing_Results/plot_timing.ipynb)


# Applications

# Advanced Features
## Detecting abnormal meteorology events
In this part, we would like to use k-means cluster technique to examine a type of climate events, called sudden stratospheric warmings (SSWs). The climatological zonal winds in the stratosphere are generally westerly and their strength increases with height. These winds form the \"polar night jet\" vortex, and can be very persistent during winters, as shown in fig(a). However, at times this zonal-mean configuration is dramatically disturbed, as shown in fig(b) and fig(c), with latitudinal temperature gradient and zonal-mean winds at the pole being reversed.

In the past, these pheonomena have been arbitrarily defined using a variety of different criteria involving winds, temperatures, and measures of the vortex shape. Using thresholds can be a powerful and useful way to understand variability, but more or less a subjective way in terms of choosing the thresholds. k-means clustering is a method of identifying different states in a completely objective manner with no preconceived notion of the groups and no preselection on the basis of known influencing factors.

k-means clustering technique is more useful than hierarchical clustering for this type of problems, because k-means clustering easily allows for uneven groups, whereas hierachical clusetering tends to determine groups of similar sizes. 

In addition, this type of problems usually involves a very large dataset with very high dimensions, e.g. more than 17,000 data points with 252 dimensions in this example, therefore a simple clustering technique such as k-means is very useful.
![Silhouette Value](Data_Analysis/figures/svalue.png)
![Temperature Anomaly](Data_Analysis/figures/T.png)
![Vortex Structure](Data_Analysis/figures/PV.png)


