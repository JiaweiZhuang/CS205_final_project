#include <stdlib.h> //for malloc
#include "make_2D_array.h"

/* For dynamically allocating 2D array in pure-C environment.
Unlke in HW2, the array here is contagious!
See:
http://stackoverflow.com/questions/33794657/how-to-pass-a-2d-array-to-a-function-in-c-when-the-array-is-formatted-like-this
http://stackoverflow.com/questions/5901476/sending-and-receiving-2d-array-over-mpi
*/
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


