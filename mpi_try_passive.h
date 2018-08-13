#include <mpi.h>

int MPIX_WIN_TRYLOCK(int lock_type, int rank, int assert, MPI_Win win, int *flag);

int MPIX_WIN_TRYUNLOCK(int rank, MPI_Win win, int *flag);
