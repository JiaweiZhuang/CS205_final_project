#include "math_util.h"

// square of the distance between x1[N_features] and x2[N_features]
float distance(int N_features,float *x1,float *x2){
    float dist=0.0;
    for (int j=0; j<N_features; j++)
        dist += (x1[j]-x2[j])*(x1[j]-x2[j]);
    return(dist);
}
