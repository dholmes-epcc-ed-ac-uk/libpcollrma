#include <mpi.h>
#include "mpi_reduce_init.h"

#include <stdlib.h>
#include <stdio.h>

int main(int argc, char **argv) {

    MPI_Init(&argc, &argv);

    long *sendbuf, *recvbuf;
    int count = 1, root = 0;
    MPIX_Request request;
    int flag = 0;
    MPI_Status status;

    double time[8];

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    sendbuf = malloc(sizeof(long));
    recvbuf = malloc(sizeof(long));

    sendbuf[0] = 3;
    recvbuf[0] = 66;


    printf("[Rank:%d] BEFORE - sendbuf[0] is %li - recvbuf[0] is %li\n", rank, sendbuf[0], recvbuf[0]);

    time[0] = MPI_Wtime();

    MPIX_REDUCE_INIT(sendbuf, recvbuf, count, MPI_LONG, MPI_SUM, root, MPI_COMM_WORLD, MPI_INFO_NULL, &request);

    time[1] = MPI_Wtime();

    MPIX_REDUCE_START(&request);

    time[2] = MPI_Wtime();

    MPIX_REDUCE_TEST(&request, &flag, &status);

    time[3] = MPI_Wtime();

    while(!flag)
        MPIX_REDUCE_TEST(&request, &flag, &status);

    time[4] = MPI_Wtime();

    MPIX_REDUCE_FREE(&request);

    time[5] = MPI_Wtime();

    printf("[Rank:%d] AFTER - sendbuf[0] is %li - recvbuf[0] is %li\n", rank, sendbuf[0], recvbuf[0]);

    printf("[Rank:%d] TIME INIT = %lf\n", rank, time[1]-time[0]);
    printf("[Rank:%d] TIME START = %lf\n", rank, time[2]-time[1]);
    printf("[Rank:%d] TIME TEST = %lf\n", rank, time[3]-time[2]);
    printf("[Rank:%d] TIME WAIT = %lf\n", rank, time[4]-time[3]);
    printf("[Rank:%d] TIME FREE = %lf\n", rank, time[5]-time[4]);
    printf("[Rank:%d] TIME Total = %lf\n", rank, time[5]-time[0]);
    printf("[Rank:%d] TIME Op = %lf\n", rank, time[4]-time[1]);

    time[6] = MPI_Wtime();
    MPI_Reduce(sendbuf, recvbuf, count, MPI_LONG, MPI_SUM, root, MPI_COMM_WORLD);
    time[7] = MPI_Wtime();

    printf("[Rank:%d] TIME MPI_Reduce = %lf\n", rank, time[7]-time[6]);

    MPI_Finalize();
}
