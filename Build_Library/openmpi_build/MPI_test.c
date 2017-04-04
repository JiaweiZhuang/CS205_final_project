#include <stdio.h> /* printf and BUFSIZ defined there */
#include <stdlib.h> /* exit defined there */
#include <mpi.h> /* all MPI-2 functions defined there */

int main(argc, argv)
        int argc;
        char *argv[];
        {
        int rank, size, length;
        char name[BUFSIZ];

        MPI_Init(&argc, &argv);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        MPI_Get_processor_name(name, &length);

        printf("%s: hello world from process %d of %d\n", name, rank, size);

        MPI_Finalize();

        exit(0);
}
