#ifndef NCDF_UTIL_H 
#define NCDF_UTIL_H

int readX(char* FILE_NAME, float*** p_X,int*** p_GUESS,
          size_t* p_N_samples,size_t* p_N_features,
          size_t* p_N_clusters,size_t* p_N_repeat );

int writeY(char* FILE_NAME, int* labels, float inert);

#endif // NCDF_UTIL_H 
