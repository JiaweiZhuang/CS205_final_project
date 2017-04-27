#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <cassert>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>

using namespace std;

struct Info {
  int     numPoints;
  int     dim;
  int     numCentroids;
  int     numRepeats;
  int    *belongs;
  float **points;
  float **centroids;
  int     thresholdLoops;
  float   thresholdFraction;
};

inline int nextPowerOfTwo(int v) {
  v--;
  v = v >> 1 | v;
  v = v >> 2 | v;
  v = v >> 4 | v;
  v = v >> 8 | v;
  v = v >> 16 | v;
  return ++v;
}

__host__ __device__ inline static float computeDist(int    dim,
                                                    int    numPoints,
                                                    int    numCentroids,
                                                    float *objects,
                                                    float *clusters,
                                                    int    objectId,
                                                    int    clusterId) {
  float res = 0;

  for (int i = 0; i < dim; i++) {
    res +=
      (objects[numPoints * i + objectId] - clusters[numCentroids * i + clusterId]) *
      (objects[numPoints * i + objectId] -
       clusters[numCentroids * i + clusterId]);
  }
  return res;
}

// Use reduction to compute the sum of an array
// Refer to
// http://developer.download.nvidia.com/compute/cuda/1.1-Beta/x86_website/projects/reduction/doc/reduction.pdf
__global__ static void reduce(int *g_idata, int l1, int l2) {
  extern __shared__ unsigned int sdata[];
  unsigned int tid = threadIdx.x;

  if (tid < l1) {
    sdata[tid] = g_idata[tid];
  } else {
    sdata[tid] = 0;
  }
  __syncthreads();

  // Parallel Reduction (l2 must be power of 2)
  for (unsigned int s = l2 / 2; s > 0; s >>= 1) {
    if (tid < s)     {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    g_idata[0] = sdata[0];
  }
}

__global__ static void find_nearest_cluster(int    dim,
                                            int    numPoints,
                                            int    numCentroids,
                                            float *objects,
                                            float *deviceClusters,
                                            int   *belongs,
                                            int   *intermediates) {
  extern __shared__ char sharedMemory[];
  unsigned char *membershipChanged = (unsigned char *)sharedMemory;
  float *clusters                  = deviceClusters;

  membershipChanged[threadIdx.x] = 0;

  int objectId = blockDim.x * blockIdx.x + threadIdx.x;

  if (objectId < numPoints)
  {
    int   index;
    float dist, min_dist;

    /*find the cluster id that has min distance to object*/
    index    = 0;
    min_dist = computeDist(dim,
                           numPoints,
                           numCentroids,
                           objects,
                           clusters,
                           objectId,
                           0);

    for (int i = 0; i < numCentroids; i++)
    {
      dist = computeDist(dim,
                         numPoints,
                         numCentroids,
                         objects,
                         clusters,
                         objectId,
                         i);

      /* no need square root */
      if (dist < min_dist)
      {
        min_dist = dist;
        index    = i;
      }
    }

    if (belongs[objectId] != index) {
      membershipChanged[threadIdx.x] = 1;
    }
    belongs[objectId] = index;

    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
      if (threadIdx.x < s)
      {
        membershipChanged[threadIdx.x] += membershipChanged[threadIdx.x + s];
      }
      __syncthreads();
    }

    if (threadIdx.x == 0)
    {
      intermediates[blockIdx.x] = membershipChanged[0];
    }
  }
}

void processData(char *fileName, Info *info) {
  float **X;
  int   **GUESS;

  int N_samples, N_features, N_clusters, N_repeat;

  // readX(FILE_NAME,&X,&GUESS,&N_samples,&N_features,&N_clusters,&N_repeat);

  // Test purpose
  N_samples  = 100;
  N_features = 9;
  N_clusters = 4;
  N_repeat   = 10;

  info->numPoints         = N_samples;
  info->dim               = N_features;
  info->numCentroids      = N_clusters;
  info->numRepeats        = N_repeat;
  info->thresholdFraction = 0.001;
  info->thresholdLoops    = 500;

  // Process data point
  X = new float*[N_samples];
  for(int i = 0; i < N_samples; i++) {
     X[i] = new float[N_features];
  }

  string str(fileName);
  ifstream file(str);
     string line1;
     int i = 0;
     while (getline(file, line1)) {
     std::istringstream iss(line1);
     int j = -1;
     for(string s; iss >> s; ) {
             if (j == -1) {
                     j++;
                     continue;
             }
             // cout << s << " ";
             X[i][j] = stof(s);
             j++;
     }
     cout << "\n";
     i++;
  }
  info->points = X;


  /* belongs: the cluster id for each data object */
  int *belongs = new int[N_samples];

  for (i = 0; i < N_samples; i++) belongs[i] = -1;
  info->belongs = belongs;
}

float** make2DArray(int x, int y) {
  float **res = (float **)malloc(x * sizeof(float *));

  for (int i = 0; i < x; i++) {
    res[i] = (float *)malloc(y * sizeof(float));
  }

  for (int i = 0; i < x; i++) {
    for (int j = 0; j < y; j++) {
      res[i][j] = 0;
    }
  }
  return res;
}

void invert2DArray(float **A, float **B, int x, int y) {
  for (int i = 0; i < x; i++) {
    for (int j = 0; j < y; j++) {
      A[i][j] = B[j][i];
    }
  }
}

void copy2DArray(float **A, float **B, int x, int y) {
  for (int i = 0; i < x; i++) {
    for (int j = 0; j < y; j++) {
      A[i][j] = B[i][j];
    }
  }
}

void cudaKmeans(Info *info) {
  // Initialization
  int numPoints         = info->numPoints;
  int dim               = info->dim;
  int numCentroids      = info->numCentroids;
  int thresholdLoops    = info->thresholdLoops;
  int thresholdFraction = info->thresholdFraction;
  int* belongs          = info->belongs;
  float **points        = info->points;
  float **centroids     = info->centroids;


  // invert
  float **iPoints = make2DArray(dim, numPoints);
  invert2DArray(iPoints, points, dim, numPoints);

  // initial guess
  float **iCentroids = make2DArray(numCentroids, numPoints);
  copy2DArray(iCentroids, points, numCentroids, numPoints);

  // centroid -> number of points
  int *pointsCount   = new int[numCentroids];
  float **iNewCentroids = make2DArray(numCentroids, numPoints);

  // Some cuda constants
  const unsigned int bthreads = 32;
  const unsigned int l1       = (numPoints + bthreads - 1) / bthreads;
  const unsigned int l2       = nextPowerOfTwo(l1);
  const unsigned int sdsize1  = bthreads * sizeof(unsigned char); // shared
                                                                  // memory size
  const unsigned int sdsize2  = l2 * sizeof(unsigned int);        // shared
                                                                  // memory size

  // Cuda device Initialization
  float *gPoints;
  float *gCentroids;
  int   *gBelongs;
  int   *tmp;
  cudaMalloc(&gPoints,    numPoints * dim * sizeof(float));
  cudaMalloc(&gCentroids, numCentroids * dim * sizeof(float));
  cudaMalloc(&gBelongs,   numPoints * sizeof(int));
  cudaMalloc(&tmp,        l2 * sizeof(unsigned int)); // For reduction
  cudaMemcpy(gBelongs,
             belongs,
             numPoints * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(gPoints,
             iPoints[0],
             numPoints * dim * sizeof(float),
             cudaMemcpyHostToDevice);

  int count = 0;
  float frac = 0.0;
  do {
    cudaMemcpy(gCentroids, iCentroids[0], numCentroids * dim * sizeof(float), cudaMemcpyHostToDevice);

    find_nearest_cluster <<< l1, bthreads, sdsize1 >>>
    (dim, numPoints, numCentroids, gPoints, gCentroids, gBelongs, tmp);
    cudaDeviceSynchronize();

    reduce <<< 1, l2, sdsize2 >>> (tmp, l1, l2);
    cudaDeviceSynchronize();

    int d;
    cudaMemcpy(&d, tmp, sizeof(int), cudaMemcpyDeviceToHost);
    frac = (float)d;

    cudaMemcpy(belongs, gBelongs, numPoints * sizeof(int), cudaMemcpyDeviceToHost);

    // Add up objects coordinates for each centroid
    for (int i = 0; i < numPoints; i++) {
      int idx = belongs[i];
      pointsCount[idx] += 1;

      for (int j = 0; j < dim; j++) {
        iNewCentroids[j][idx] += points[i][j];
      }
    }

    // Update centroids
    for (int i = 0; i < numCentroids; i++) {
      for (int j = 0; j < dim; j++) {
        if (pointsCount[i] > 0) {
          iCentroids[j][i] = iNewCentroids[j][i] / pointsCount[i];
        }
        iNewCentroids[j][i] = 0.0;
      }
      pointsCount[i] = 0;
    }
    frac /= numPoints;
    count++;
  } while (frac > thresholdFraction && count < 500);

  invert2DArray(centroids, iCentroids, numCentroids, dim);

}

int main(int argc, char *argv[]) {
  Info *info     = new Info;
  char *fileName = argv[1];

  processData(fileName, info);
  cudaKmeans(info);
}
