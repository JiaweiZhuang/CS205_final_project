#include "math_util.h"

// square of the distance between x1[N_features] and x2[N_features]
float distance(int N_features,float *x1,float *x2){
    float dist=0.0;
    for (int j=0; j<N_features; j++)
        dist += (x1[j]-x2[j])*(x1[j]-x2[j]);
    return(dist);
}

float correlation(int N_features,float *x,float *y){
   float xsum=0.0,ysum=0.0,xysum=0.0,xsqr_sum=0.0,ysqr_sum=0.0;
   for (int j = 0; j<N_features; j++) {
            xsum = xsum + x[i];
            ysum = ysum + y[i];
            xysum = xysum + x[i] * y[i];
            xsqr_sum = xsqr_sum + x[i] * x[i];
            ysqr_sum = ysqr_sum + y[i] * y[i];
    }

    float num = ((N_features * xysum) - (xsum * ysum));
    float deno = ((N_features * xsqr_sum - xsum * xsum)* (N_features * ysqr_sum - ysum * ysum));
    float coeff = num / sqrt(deno);
    return(coeff);
}
