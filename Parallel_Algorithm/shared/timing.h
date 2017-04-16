#include <sys/time.h>
#include <stdlib.h>

inline double seconds()
{
    struct timeval tp;
    int i = gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}
