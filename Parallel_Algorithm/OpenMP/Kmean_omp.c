#include "../shared/timing.h" //for timer seconds()
#include "../shared/make_2D_array.h"
#include "../shared/ncdf_util.h"
#include "../shared/math_util.h"
#include <stdio.h>
#include <stdlib.h>
#include <float.h> //for FLT_MAX

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
    size_t N_samples,N_features,N_clusters,N_repeat;
    //i for samples; j for features; k for clusters (typically)
    int i,j,k;
    int k_best,initial_idx;
    float** X;
    int** GUESS;
    float dist,dist_min,dist_sum_old,dist_sum_new,inert_best=FLT_MAX;
    
    // get input data and its size
    double iStart1 = seconds();
    readX(FILE_NAME,&X,&GUESS,&N_samples,&N_features,&N_clusters,&N_repeat);
    double iElaps1 = seconds() - iStart1;

    // each data point belongs to which cluster
    // values range from 0 to N_cluster-1
    int* labels = (int *)malloc(N_samples*sizeof(int));
    int* labels_best = (int *)malloc(N_samples*sizeof(int));

    // The position of each cluster center.
    // Two arrays are needed as we are calculating the distance to the
    // old centers and accumulating the new centers in the same iteration.
    float** old_cluster_centers = Make2DFloatArray(N_clusters,N_features);
    float** new_cluster_centers = Make2DFloatArray(N_clusters,N_features);

    // how many data points in the cluster
    // needed by calculating the average position of data points in each cluster
    int* cluster_sizes = (int *)malloc(N_clusters*sizeof(int)); 

    /*
    ======================================================
    ----------------  Kmean stepping ---------------------
    ======================================================
    */
    printf("=====Applying K-mean======\n");
   
    // record timing results 
    double iStart2,iElaps2;
    double iStart3a,iStart3b,iStart3c;
    double iElaps3a=0,iElaps3b=0,iElaps3c=0;

    /* Run the K-mean algorithm for N_repeat times with 
     * different starting points
     */
    iStart2 = seconds();
    for (int i_repeat=0; i_repeat < N_repeat; i_repeat++){

    // guess initial centers
    for (k=0; k<N_clusters; k++){
        cluster_sizes[k] = 0; // for accumulating 
        // the index of data points as the initial guess for cluster centers
        initial_idx = GUESS[i_repeat][k]; 
        for (j=0; j<N_features; j++){
            old_cluster_centers[k][j]=X[initial_idx][j];
            //set the "new" array to 0 for accumulating
            new_cluster_centers[k][j] = 0.0;
    }
    }

    // core K-mean stepping (Expectation-Maximization) begins here!!
    int i_iter = 0;//record iteration counts
    dist_sum_new = 0.0;//prevent the firt iteration error
    do {
    i_iter++;
    dist_sum_old = dist_sum_new; 
    dist_sum_new = 0.0;

    // E-Step: assign points to the nearest cluster center
    iStart3a = seconds();
    #pragma omp parallel default(shared) \
            private(i,j,k,k_best,dist,dist_min) \
            reduction(+:dist_sum_new)
    {
    #pragma omp for schedule(static)
    for (i = 0; i < N_samples; i++) {
        k_best = 0;//assume cluster no.0 is the nearest
        dist_min = distance(N_features, X[i], old_cluster_centers[k_best]); 
        for (k = 1; k < N_clusters; k++){
            dist = distance(N_features, X[i], old_cluster_centers[k]); 
            if (dist < dist_min){
                dist_min = dist;
                k_best = k; 
            }
        }
       labels[i] = k_best;
       dist_sum_new += dist_min;
    } // end of omp for loop
    } // end of omp parallel region
    iElaps3a += (seconds()-iStart3a);

    // M-Step first half: set the cluster centers to the mean
    iStart3b = seconds();
    for (i = 0; i < N_samples; i++) {
        k_best = labels[i];
        cluster_sizes[k_best]++; // add one more points to this cluster
        // As the total number of samples in each cluster is not known yet,
        // here we are just calculating the sum, not the mean.
        for (j=0; j<N_features; j++)
            new_cluster_centers[k_best][j] += X[i][j]; 
    } // end of M-Step first half
    iElaps3b += (seconds()-iStart3b);

    iStart3c = seconds();
    // M-Step second half: convert the sum to the mean
    for (k=0; k<N_clusters; k++) {
            for (j=0; j<N_features; j++) {

                if (cluster_sizes[k] > 0) //avoid divide-by-zero error
                    // sum -> mean
                    old_cluster_centers[k][j] = new_cluster_centers[k][j] / cluster_sizes[k];
               
               new_cluster_centers[k][j] = 0.0;//for the next iteration
            }
            cluster_sizes[k] = 0;//for the next iteration
    } // end of M-Step second half
    iElaps3c += (seconds()-iStart3c);

    } while( i_iter==1 || ((dist_sum_old - dist_sum_new > TOL)&&i_iter<MAX_ITER) ); 
    //end of K-mean stepping

    //printf("Final inertia: %f, iteration: %d \n",dist_sum_new,i_iter);

    // record the best results
    if (dist_sum_new < inert_best) {
        inert_best = dist_sum_new;
        for (i = 0; i < N_samples; i++)
            labels_best[i] = labels[i];
    }

    } //end of one repeated run
    iElaps2 = seconds() - iStart2;

    /*
    ======================================================
    ----------------  Finalization ---------------------
    ======================================================
    */

    // write data back to NetCDF file
    writeY(FILE_NAME,labels_best, inert_best);

    // print summary
    printf("Best inertia: %f \n",inert_best);
    printf("Kmean total time use (ms): %f \n", iElaps2*1000.0);
    printf("E-step time use (ms): %f \n", iElaps3a*1000.0);
    printf("M-step-1st-half time use (ms): %f \n", iElaps3b*1000.0);
    printf("M-step-2nd-half time use (ms): %f \n", iElaps3c*1000.0);
    printf("I/O time use (ms): %f \n", iElaps1*1000.0);

    return 0;
}
