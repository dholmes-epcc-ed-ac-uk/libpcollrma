#include <mpi.h>

typedef struct pcoll_rma_sched_t {
    const void *origin_addr;
    MPI_Count origin_count;
    MPI_Datatype origin_datatype;
    int target_rank;
    MPI_Aint target_disp;
    MPI_Count target_count;
    MPI_Datatype target_datatype;
    MPI_Op op;
} pcoll_rma_sched;

typedef struct MPIX_Request_t {
    unsigned int isactive   : 1;
    unsigned int iamroot    : 1;
    unsigned int isinplace  : 1;
    unsigned int isstarting : 1;
    unsigned int toggle     : 1;
    unsigned int            : 0;
    int          myrank;
    MPI_Comm     comm;
    MPI_Request  req_ib;
    MPI_Status   status;
    MPI_Win      win[2];
    struct pcoll_rma_sched_t
                 sched;
    enum {isRoot=-1, isEarly, isExposed, isContention, isUnlocked, isBlocked, isComplete}
                 state;
} MPIX_Request;

int MPIX_REDUCE_INIT(const void* sendbuf, void* recvbuf,
                     int count, MPI_Datatype datatype,
                     MPI_Op op, int root,
                     MPI_Comm comm, MPI_Info info,
                     MPIX_Request *request);

int MPIX_REDUCE_START(MPIX_Request *const request);

int MPIX_REDUCE_TEST(MPIX_Request *const request, int *const flag, MPI_Status *const status);

int MPIX_REDUCE_REQUEST_GET_STATUS(MPIX_Request *const request, int *const flag, MPI_Status *const status);

int MPIX_REDUCE_FREE(MPIX_Request *request);

