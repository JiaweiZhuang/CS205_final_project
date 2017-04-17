//#include "../shared/timing.h" //for timer seconds()
#include <stdio.h>
#include <stdlib.h>
#include <float.h> //for FLT_MAX
#include <mpi.h>
#include "../shared/make_2D_array.h"
#include "../shared/ncdf_util.h"
#include "../shared/math_util.h"

/* This is the name of the data file we will read. */
#define FILE_NAME "../test_data/Blobs_smp20000_fea30_cls8.nc"
#define TOL 0.0001 
#define MAX_ITER 100 

int main() {

    /*
    ======================================================
    ----------------  Initialization ---------------------
    ======================================================
    */

    int rank, size;
    MPI_Init(NULL,NULL);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    //printf("hello world from process %d of %d\n", rank, size);

    size_t N_samples_all,N_samples,N_features,N_clusters,N_repeat;
    //i for samples; j for features; k for clusters (typically)
    int i,j,k;
    int k_best,initial_idx;
    float** X;
    float** X_all; //only master node holds the full data
    int** GUESS;
    float dist,dist_min,dist_sum_old,dist_sum_new,inert_best=FLT_MAX;

    /*
    ======================================================
    -- Read data by master node and distribute over processes --
    ======================================================
    */
    
    double iStart1 = MPI_Wtime(); 
    // let master core read data and broadcast to other cores

    if (rank == 0){
    // get input data and its size
    readX(FILE_NAME,&X_all,&GUESS,&N_samples_all,&N_features,&N_clusters,&N_repeat);
    }
    else{
    /*
    MPI_Scatter needs to access *X_all in all processes
    For non-root, we need to assign NULL for prevent memory error
    */
    float* dummy_for_X_all=NULL;
    X_all = &dummy_for_X_all;
    }

    MPI_Bcast(&N_samples_all,1,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Bcast(&N_features,1,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Bcast(&N_clusters,1,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Bcast(&N_repeat,1,MPI_INT,0,MPI_COMM_WORLD);
    //printf("%d: %d,%d,%d,%d\n",rank,N_samples_all,N_features,N_clusters,N_repeat);

    /*
    Assume N_sample_all is divisible by size for now!
    TBD: use MPI_Scatterv to handle arbitrary size
    */

    //convert to int to prevent error from unsigned/signed
    N_samples = (int)N_samples_all / size; 
    // printf("%d, Local samples: %d \n",rank,N_samples);

    if (rank==0){
        printf("Last element in global array: %d: %f \n",X_all[N_samples_all-1][N_features-1]);
    }

    X = Make2DFloatArray(N_samples,N_features);
    MPI_Scatter(*X_all, N_samples*N_features, MPI_FLOAT, *X,
           N_samples*N_features, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // check scattered results
    if (rank==size-1){
        //convert to int to prevent error from unsigned - signed
        printf("Last element after sacattering %d: %f \n",rank,X[(int)N_samples-1][(int)N_features-1]);
    }

    double iElaps1 = MPI_Wtime() - iStart1;

    /*
    ======================================================
    -------  Continue to initialize variables
    ======================================================
    */

    // each data point belongs to which cluster
    // values range from 0 to N_cluster-1
    int* labels = (int *)malloc(N_samples*sizeof(int));
    // int* labels_best = (int *)malloc(N_samples*sizeof(int));

    // The position of each cluster center.
    // Two arrays are needed as we are calculating the distance to the
    // old centers and accumulating the new centers in the same iteration.
    float** old_cluster_centers = Make2DFloatArray(N_clusters,N_features);
    float** new_cluster_centers = Make2DFloatArray(N_clusters,N_features);

    // how many data points in the cluster
    // needed by calculating the average position of data points in each cluster
    int* cluster_sizes = (int *)malloc(N_clusters*sizeof(int));
    // cluster_sizes[0] = 0; //why this will fail for rank1?

    /*
    ======================================================
    ----------------  Kmean stepping ---------------------
    ======================================================
    */
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0)
        printf("=====Applying K-mean======\n");

    int i_repeat = 0;

    // guess initial centers
    if (rank==0) {
    // only master node holds the full data X_all and the initial GUESS
    for (k=0; k<N_clusters; k++){
        cluster_sizes[k] = 0; // for accumulating
        // the index of data points as the initial guess for cluster centers
        initial_idx = GUESS[i_repeat][k];
        for (j=0; j<N_features; j++){
            old_cluster_centers[k][j]=X_all[initial_idx][j];
            //set the "new" array to 0 for accumulating
            new_cluster_centers[k][j] = 0.0;
            }
        }
    }
    if (rank==1){
    // initialize other nodes
    for (k=0; k<N_clusters; k++){
        //cluster_sizes[k] = 0; 
        for (j=0; j<N_features; j++){
            //new_cluster_centers[k][j] = 0.0;
            }
        }
    }

    if(rank==0)
        printf("master node: %f \n",old_cluster_centers[(int)N_clusters-1][(int)N_features-1]);
    MPI_Bcast(*old_cluster_centers,N_clusters*N_features,MPI_FLOAT,0,MPI_COMM_WORLD);

    // check broadcast results
    printf("%d : %f \n",rank,old_cluster_centers[(int)N_clusters-1][(int)N_features-1]);

    /*
    ======================================================
    ----------------  Finalization ---------------------
    ======================================================
    */

    // write data back to NetCDF file
    // writeY(FILE_NAME,labels_best, inert_best);


    /* get the max timing measured among all processes */
    double iElaps1_max;
    MPI_Reduce(&iElaps1, &iElaps1_max, 1, MPI_DOUBLE,
                MPI_MAX, 0, MPI_COMM_WORLD);

    // print summary
    if (rank == 0){
    printf("Best inertia: %f \n",inert_best);
    /*
    printf("Kmean total time use (ms): %f \n", iElaps2*1000.0);
    printf("E-step time use (ms): %f \n", iElaps3a*1000.0);
    printf("M-step-1st-half time use (ms): %f \n", iElaps3b*1000.0);
    printf("M-step-2nd-half time use (ms): %f \n", iElaps3c*1000.0);
    */
    printf("I/O time use (ms): %f \n", iElaps1_max*1000.0);
    }

    MPI_Finalize();

    return 0;
}
