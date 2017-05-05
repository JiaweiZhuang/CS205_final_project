do{
    dist_sum_old = dist_sum_new;
    dist_sum_new = 0.0;
    // E-Step: assign points to the nearest cluster center
    for (i = 0; i < N_samples; i++) {
        k_best = 0;
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
    }

    // M-Step: set the cluster centers to the mean
    // M-Step first half: As the total number of samples in each cluster is not known yet,here we are just calculating the sum, not the mean.
    for (i = 0; i < N_samples; i++) {
        k_best = labels[i];
        cluster_sizes[k_best]++; 
        for (j=0; j<N_features; j++)
            new_cluster_centers[k_best][j] += X[i][j]; 
    } 
    // M-Step second half: convert the sum to the mean
    for (k=0; k<N_clusters; k++) {
    for (j=0; j<N_features; j++) {
        old_cluster_centers[k][j] = new_cluster_centers[k][j] / cluster_sizes[k];
    }
    }
} while(dist_sum_old - dist_sum_new > TOL)


