#include <netcdf.h>
int main()
{
   int ncid;
   if (nc_create("tmp.nc", NC_NETCDF4, &ncid))
      return 1;
   if (nc_close(ncid))
      return 2;
   return 0;
}


