#include "mpi_try_passive.h"

int MPIX_WIN_TRYLOCK(int lock_type, int rank, int assert, MPI_Win win, int *flag) {
    int retVal = MPI_SUCCESS;
    retVal = MPI_Win_lock(lock_type, rank, assert, win);
    *flag = MPI_SUCCESS==retVal ? 1 : 0;
    return retVal;
}

int MPIX_WIN_TRYUNLOCK(int rank, MPI_Win win, int *flag) {
    int retVal = MPI_SUCCESS;
    retVal = MPI_Win_unlock(rank, win);
    *flag = MPI_SUCCESS==retVal ? 1 : 0;
    return retVal;
}
