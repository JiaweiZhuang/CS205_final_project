#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <time.h>
#include <sys/time.h>
extern "C" {
  #include <netcdf.h>
}

using namespace std;

// #define FAKE_DATA "../test_data/Blobs_smp20000_fea30_cls8.nc"
#define ERRCODE 2
#define ERR(e) {printf("Error: %s\n", nc_strerror(e)); exit(ERRCODE);}

double iStart1, iStart2, iStart3a, iStart3b, iStart4a, iStart4b, iStart4c, iStart4d, iStart5;
double iElaps1=0, iElaps2=0, iElaps3a=0, iElaps3b=0, iElaps4=0, iElaps5=0;
// Hold configurations for Kmeans
struct Info {
  int     numPoints;
  int     dim;
  int     numCentroids;
  int     numRepeats;
  int    *belongs;
  float **points;
  float **centroids;
  float **guess;
  int     thresholdLoops;
  float   thresholdFraction;
  int     threadPerBlock;
};

// ************************** Utils ************************** //

float** Make2DFloatArray(int rows, int cols) {
    float *data = (float *)malloc(rows*cols*sizeof(float));
    float **array= (float **)malloc(rows*sizeof(float*));
    for (int i=0; i<rows; i++)
        array[i] = &(data[cols*i]);

    return array;
}
int** Make2DIntArray(int rows, int cols) {
    int *data = (int *)malloc(rows*cols*sizeof(int));
    int **array= (int **)malloc(rows*sizeof(int*));
    for (int i=0; i<rows; i++)
        array[i] = &(data[cols*i]);

    return array;
}

int readX(char* FILE_NAME, float*** p_X,int*** p_GUESS, int* p_N_samples,int* p_N_features, int* p_N_clusters, int* p_N_repeat) {
    int ncid, varid,dimid;
    int retval;
    size_t N_temp;

    // printf("Reading data...\n");

    /* Open the file. NC_NOWRITE tells netCDF we want read-only access
     * to the file.*/
    if ((retval = nc_open(FILE_NAME, NC_NOWRITE, &ncid)))
       ERR(retval);

    /* Get the size of the data for dynamical allocation*/
    nc_inq_dimid(ncid,"N_samples",&dimid);
    nc_inq_dimlen(ncid,dimid,&N_temp);
    *p_N_samples = (int)N_temp;
    // printf("Number of samples: %d \n",*p_N_samples);

    nc_inq_dimid(ncid,"N_features",&dimid);
    nc_inq_dimlen(ncid,dimid,&N_temp);
    *p_N_features = (int)N_temp;
    // printf("Number of features: %d \n",*p_N_features);

    nc_inq_dimid(ncid,"N_clusters",&dimid);
    nc_inq_dimlen(ncid,dimid,&N_temp);
    *p_N_clusters = (int)N_temp;
    // printf("Number of clusters: %d \n",*p_N_clusters);

    nc_inq_dimid(ncid,"N_repeat",&dimid);
    nc_inq_dimlen(ncid,dimid,&N_temp);
    *p_N_repeat = (int)N_temp;
    // printf("Number of repeated runs: %d \n",*p_N_repeat);

     /* Get the varid of the data variable, based on its name. */
    if ((retval = nc_inq_varid(ncid, "X", &varid)))
       ERR(retval);
    /* Read the data. */
    *p_X = Make2DFloatArray(*p_N_samples,*p_N_features);
    if ((retval = nc_get_var_float(ncid, varid, (*p_X)[0])))
       ERR(retval);

     /* Initial Guess*/
    if ((retval = nc_inq_varid(ncid, "GUESS", &varid)))
       ERR(retval);
    *p_GUESS = Make2DIntArray(*p_N_repeat,*p_N_clusters);
    if ((retval = nc_get_var_int(ncid, varid, (*p_GUESS)[0])))
       ERR(retval);

    /*close the netcdf file*/
    if ((retval = nc_close(ncid) ))
       ERR(retval);

    // printf("Reading data finished. \n");

    return 0;
 }


double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp,NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

static inline int nextPowerOfTwo(int v) {
  int res = v;
  for (int i = 1; i <= 16; i *= 2) {
    res |= res >> i;
  }
  return res + 1;
}

float** make2DArray(int x, int y) {
  float **res = (float **)malloc(x * sizeof(float *));

  // for (int i = 0; i < x; i++) {
  //   res[i] = (float *)malloc(y * sizeof(float));
  // }
  res[0] = (float *)malloc(x * y * sizeof(float));
  for (size_t i = 1; i < x; i++) res[i] = res[i-1] + y;
  for (size_t i = 0; i < x; i++) {
    for (size_t j = 0; j < y; j++) {
      res[i][j] = 0.0;
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

// ************************** Utils ************************** //

__host__ __device__ inline static float
    computeDist(Info* info, int pointId, int centroidId, int distType, float *gPoints, float *gCentroids) {
  float res = 0;
  if (distType == 0) {
    for (int i = 0; i < info->dim; i++) {
      res +=
        (gPoints[i * (info->numPoints) + pointId] - gCentroids[i * (info->numCentroids) + centroidId]) *
        (gPoints[i * (info->numPoints) + pointId] - gCentroids[i * (info->numCentroids) + centroidId]);
    }
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

__global__ static void nearestCentroid(int *blockResult, int *gBelongs, float *gPoints, float *gCentroids, Info *gInfo) {

  int pointId = blockDim.x * blockIdx.x + threadIdx.x;
  if (pointId >= (gInfo->numPoints)) return;

  // For test on test.txt
  // printf("Thread: %d - %.2f,  %.2f, %.2f,  %.2f \n", pointId, gCentroids[0], gCentroids[1], gCentroids[2], gCentroids[3]);
  // printf("Thread: %d - %.2f,  %.2f, %.2f,  %.2f, %.2f,  %.2f, %.2f,  %.2f \n",
  //          pointId, gPoints[0], gPoints[1], gPoints[2], gPoints[3], gPoints[4], gPoints[5], gPoints[6], gPoints[7]);

  // Get the minimum distance
  float mDist = computeDist(gInfo, pointId, 0, 0, gPoints, gCentroids);

  int tmpIdx = 0;
  int numCentroids = gInfo->numCentroids;
  for (int i = 0; i < numCentroids; i++) {
    float tmpDist = computeDist(gInfo, pointId, i, 0, gPoints, gCentroids);
    if (tmpDist < mDist) {
      mDist  = tmpDist;
      tmpIdx = i;
    }
  }

  // use reduction to add the total number of changes (change from one centroid to another) in this block
  extern __shared__ int sdata2[];
  sdata2[threadIdx.x] = 0;
  if (gBelongs[pointId] != tmpIdx) {
    sdata2[threadIdx.x] = 1;
  }
  gBelongs[pointId] = tmpIdx;
  __syncthreads();

  // Reduction
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      sdata2[threadIdx.x] += sdata2[threadIdx.x + s];
    }
    __syncthreads();
  }

  // Put the sum to the location corresbonding to current block
  if (threadIdx.x == 0) {
    blockResult[blockIdx.x] = sdata2[0];
  }
}

void processData(char *fileName, Info *info, int i_repeat) {
  float **X;
  int   **GUESS;

  int N_samples, N_features, N_clusters, N_repeat;

  readX(fileName,&X,&GUESS,&N_samples,&N_features,&N_clusters,&N_repeat);

  // cout << N_samples << "," <<  N_features << ","  <<  N_clusters << "," << N_repeat << '\n';

  // Test purpose
  // N_samples  = 4;
  // N_features = 2;
  // N_clusters = 2;
  // N_repeat   = 1;

  info->numPoints         = N_samples;
  info->dim               = N_features;
  info->numCentroids      = N_clusters;
  info->numRepeats        = N_repeat;
  info->thresholdFraction = 0.001;
  info->thresholdLoops    = 200;
  info->points            = X;

  float **guess = make2DArray(N_clusters, N_features);
  for (int k=0; k<N_clusters; k++){
       int initial_idx = GUESS[i_repeat][k];
       for (int j=0; j<N_features; j++){
           guess[k][j]=X[initial_idx][j];
       }
   }
   info->guess = guess;

  /* belongs: the cluster id for each data object */
  int *belongs = new int[N_samples];
  for (int i = 0; i < N_samples; i++) belongs[i] = -1;
  info->belongs = belongs;
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
  float **guess         = info->guess;
  int threadPerBlock    = info->threadPerBlock;

  iStart4d = cpuSecond();

  // invert (transpose matrix)
  float **iPoints = make2DArray(dim, numPoints);
  invert2DArray(iPoints, points, dim, numPoints);

  // initial guess
  float **iCentroids = make2DArray(dim, numCentroids);
  // copy2DArray(iCentroids, iPoints, dim, numCentroids);
  invert2DArray(iCentroids, guess, dim, numCentroids);

  // centroid -> number of points
  int *pointsCount   = new int[numCentroids];
  float **iNewCentroids = make2DArray(dim, numCentroids);

  iElaps4 += cpuSecond() - iStart4d;

  // Some cuda constants
  const unsigned int bthreads = threadPerBlock;
  const unsigned int l1       = (numPoints + bthreads - 1) / bthreads;
  const unsigned int l2       = nextPowerOfTwo(l1);
  const unsigned int sdsize2  = bthreads * sizeof(unsigned int); // shared memory size for sdata2
  const unsigned int sdsize1  = l2 * sizeof(unsigned int); // shared memory size for sdata1

  // Cuda device Initialization
  float *gPoints;
  float *gCentroids;
  int   *gBelongs;
  Info   *gInfo;
  int   *tmp;

  // Data transfer
  iStart4a = cpuSecond();
  cudaMalloc(&gPoints,    numPoints * dim * sizeof(float));
  cudaMalloc(&gCentroids, numCentroids * dim * sizeof(float));
  cudaMalloc(&gBelongs,   numPoints * sizeof(int));
  cudaMalloc((void**)&gInfo,   sizeof(Info));
  cudaMalloc(&tmp,        l2 * sizeof(unsigned int)); // For reduction
  cudaMemcpy(gBelongs,
             belongs,
             numPoints * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(gPoints,
             iPoints[0],
             numPoints * dim * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(gInfo,info,sizeof(Info),cudaMemcpyHostToDevice);

  iElaps4 += cpuSecond() - iStart4a;

  int count = 0;
  float frac = 1.0;

  while (count < thresholdLoops) {
    iStart4b = cpuSecond();
    cudaMemcpy(gCentroids, iCentroids[0], dim * numCentroids * sizeof(float), cudaMemcpyHostToDevice);
    iElaps4 += cpuSecond() - iStart4b;

    // E-Step: assign points to the nearest cluster center
    iStart2 = cpuSecond();
    // nearestCentroid<<<l1, bthreads, sdsize2>>>(dim, numPoints, numCentroids, gPoints, gCentroids, gBelongs, tmp);
    nearestCentroid<<<l1, bthreads, sdsize2>>>(tmp, gBelongs, gPoints, gCentroids, gInfo);
    cudaDeviceSynchronize();
    iElaps2 += (cpuSecond() - iStart2);

    // Update belongs
    iStart4c = cpuSecond();
    cudaMemcpy(belongs, gBelongs, numPoints * sizeof(int), cudaMemcpyDeviceToHost);
    iElaps4 += cpuSecond() - iStart4c;

    // M-Step first half: set the cluster centers to the mean
    iStart3a = cpuSecond();

    // Clear the two temp variables
    for (int i = 0; i < numCentroids; i++) {
      pointsCount[i] = 0;
      for (int j = 0; j < dim; j++) {
        iNewCentroids[j][i] = 0.0;
      }
    }

    // Add up points in each centroid
    for (int i = 0; i < numPoints; i++) {
      int idx = belongs[i];
      pointsCount[idx] += 1;
      for (int j = 0; j < dim; j++) {
        iNewCentroids[j][idx] += points[i][j];
      }
    }
    iElaps3a += cpuSecond() - iStart3a;

    // M-Step second half: convert the sum to the mean
    // Update to new centroids
    iStart3b = cpuSecond();
    for (int i = 0; i < numCentroids; i++) {
      for (int j = 0; j < dim; j++) {
        if (pointsCount[i] > 0) {
          iCentroids[j][i] = iNewCentroids[j][i] / pointsCount[i];
        }
      }
    }
    iElaps3b += cpuSecond() - iStart3b;

    // Check convergence
    iStart5 = cpuSecond();

    // Check if too few number of points change their centroids
    reduce <<<1, l2, sdsize1>>>(tmp, l1, l2);
    cudaDeviceSynchronize();
    int tmpFloat;
    cudaMemcpy(&tmpFloat, tmp, sizeof(int), cudaMemcpyDeviceToHost);
    frac = (float)tmpFloat / numPoints;
    // cout << "Iteration: " << count << "," << frac << "," << tmpFloat  << "\n";
    count++;
    if (frac <= thresholdFraction) break;

    iElaps5 += cpuSecond() - iStart5;

  }

  iStart4d = cpuSecond();
  centroids = make2DArray(numCentroids, dim);
  invert2DArray(centroids, iCentroids, numCentroids, dim);
  info->centroids = centroids;
  iElaps4 += cpuSecond() - iStart4d;

  // Free device memory
  cudaFree(gPoints);
  cudaFree(gCentroids);
  cudaFree(gBelongs);
  cudaFree(tmp);

}

int main(int argc, char *argv[]) {
  Info *info     = new Info;
  info->threadPerBlock = atoi(argv[1]);
  char *fileName = argv[2];
  processData(fileName, info, 0);

  printf("Number of samples: %d \n",info->numPoints);
  printf("Number of features: %d \n", info->dim);
  printf("Number of clusters: %d \n", info->numCentroids);
  printf("Number of repeated runs: %d \n", info->numRepeats);
  for (int i = 0; i < info->numRepeats; i++) {
    // cout << "====== Begin Loop " << i << " ======\n";
    iStart1 = cpuSecond();
    cudaKmeans(info);
    iElaps1 += cpuSecond() - iStart1;

    // cout << "Ref: " << info->centroids[0][0] << "\n";
    // cout << "====== End of Loop " << i << " ======\n";
    // break;

    // Reload info
    delete(info);
    if (i + 1== info->numRepeats) break;
    info     = new Info;
    info->threadPerBlock = atoi(argv[1]);
    processData(fileName, info, i+1);
  }


  cout << "Total time: " << iElaps1*1000 << "\n";
  cout << "E-step time use (ms): " << iElaps2*1000 << "\n";
  cout << "M-step-1st-half time use (ms): " << iElaps3a*1000 << "\n";
  cout << "M-step-2nd-half time use (ms): " << iElaps3b*1000 << "\n";
  cout << "Cuda Data IO (ms): " << iElaps4*1000 << "\n";
  cout << "Check Convergence (ms): " << iElaps5*1000 << "\n";
}
