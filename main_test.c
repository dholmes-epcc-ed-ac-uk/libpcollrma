#include <mpi.h>
#include "mpi_reduce_init.h"

#include <stdlib.h>
#include <stdio.h>

enum times {
    start_time,
    before_INIT = start_time,
    after_INIT,
    before_START = after_INIT,
    after_START,
    before_TEST = after_START,
    after_TEST,
    before_WAIT = after_TEST,
    after_WAIT,
    before_FREE = after_WAIT,
    after_FREE,
    before_MPI,
    after_MPI,
    end_time,
    num_time = end_time
};
const char *time_names[num_time] = {"before_INIT",
                                    "between_INIT_and_START",
                                    "between_START_and_TEST",
                                    "between_TEST_and_WAIT",
                                    "between_WAIT_and_FREE",
                                    "after_FREE",
                                    "before_MPI", "after_MPI"};

enum stats{
    start_stat,
    First = start_stat,
    MIN,
    Quartile_1st,
    MEDIAN,
    Quartile_3rd,
    MAX,
    MEAN,
    end_stat,
    num_stat = end_stat
};

enum intervals {
    start_interval,
    interval_INIT = start_interval,
    interval_START,
    interval_TEST,
    interval_WAIT,
    interval_FREE,
    interval_TOTAL,
    interval_MPI,
    end_interval,
    num_interval = end_interval
};

const int num_trials = 100;

double times[num_trials][num_time];
double stats[num_stat][num_interval];

double interval(int t, enum intervals i) {
    double interval = 0.0;
    switch(i) {
      case interval_INIT:
        interval = times[t][after_INIT] - times[t][before_INIT];
        break;
      case interval_START:
        interval = times[t][after_START] - times[t][before_START];
        break;
      case interval_TEST:
        interval = times[t][after_TEST] - times[t][before_TEST];
        break;
      case interval_WAIT:
        interval = times[t][after_WAIT] - times[t][before_WAIT];
        break;
      case interval_FREE:
        interval = times[t][after_FREE] - times[t][before_FREE];
        break;
      case interval_TOTAL:
        interval = times[t][after_FREE] - times[t][before_INIT];
        break;
      case interval_MPI:
        interval = times[t][after_MPI] - times[t][before_MPI];
        break;
      default:
        break;
    }
    return interval;
}

const char *interval_names[num_interval] = {"INIT", "START", "TEST", "WAIT", "FREE", "TOTAL", "MPI"};

void OUTPUT(int rank, enum intervals i) {
    printf("[Rank:%d] STATS for %s 1st=%lf (MIN=%lf, MEAN=%lf, MAX=%lf)\n",
           rank, interval_names[i],
           stats[First][i],
           stats[MIN][i],
           stats[MEAN][i],
           stats[MAX][i]);
}

void DUMP_TIMES(int rank, int t) {
    if (!rank & !t) {
        printf("Rank,Trial,Time,Name,Value\n");
    }
    for (enum times n=start_time;n<num_time;++n) {
        printf("%d,%d,%d,%s,%lf\n",
               rank, t, n, time_names[n],
               times[t][n]);
    }
}

void DUMP_INTERVALS(int rank, int t) {
    if (!rank & !t) {
        printf("Rank,Trial,Interval,Name,Value\n");
    }
    for (enum intervals i=start_interval;i<num_interval;++i) {
        printf("%d,%d,%d,%s,%lf\n",
               rank, t, i, interval_names[i],
               interval(t, i));
    }
}

int main(int argc, char **argv) {

    MPIX_Request request;
    MPI_Status status;
    long *sendbuf, *recvbuf;
    int rank, count = 1, root = 0, flag = 0;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    sendbuf = malloc(sizeof(long));
    recvbuf = malloc(sizeof(long));

    sendbuf[0] = 3;
    recvbuf[0] = 66;

    printf("[Rank:%d] BEFORE - sendbuf[0] is %li - recvbuf[0] is %li\n", rank, sendbuf[0], recvbuf[0]);

    for (int t=0;t<num_trials;++t) {
        times[t][before_MPI] = MPI_Wtime();
        MPI_Reduce(sendbuf, recvbuf, count, MPI_LONG, MPI_SUM, root, MPI_COMM_WORLD);
        times[t][after_MPI] = MPI_Wtime();
    }

    printf("[Rank:%d] AFTER - sendbuf[0] is %li - recvbuf[0] is %li\n", rank, sendbuf[0], recvbuf[0]);

    sendbuf[0] = 3;
    recvbuf[0] = 66;

    times[0][before_INIT] = MPI_Wtime();

    MPIX_REDUCE_INIT(sendbuf, recvbuf, count, MPI_LONG, MPI_SUM, root, MPI_COMM_WORLD, MPI_INFO_NULL, &request);

    for (int t=0;t<num_trials;++t) {
        times[t][before_START] = MPI_Wtime();

        MPIX_REDUCE_START(&request);

        times[t][before_TEST] = MPI_Wtime();

        MPIX_REDUCE_TEST(&request, &flag, &status);

        times[t][after_TEST] = MPI_Wtime();

        while(!flag) // busy wait
            MPIX_REDUCE_TEST(&request, &flag, &status);

        times[t][after_WAIT] = MPI_Wtime();
    }

    MPIX_REDUCE_FREE(&request);

    times[num_trials-1][after_FREE] = MPI_Wtime();

    printf("[Rank:%d] AFTER - sendbuf[0] is %li - recvbuf[0] is %li\n", rank, sendbuf[0], recvbuf[0]);

    // INIT time for non-first trials is zero
    // FREE time for non-final trials is zero
    times[0][after_FREE] = times[0][before_FREE];
    for (int t=1;t<num_trials-1;++t) {
        times[t][before_INIT] = times[t][after_INIT];
        times[t][after_FREE] = times[t][before_FREE];
    }
    times[num_trials-1][before_INIT] = times[num_trials-1][after_INIT];

    // compute aggregation stats
    for (enum intervals i=start_interval;i<num_interval;++i) {
        stats[MIN][i] = 0.0;
        stats[MEAN][i] = 0.0;
        stats[MAX][i] = 0.0;
        for (int t=1;t<num_trials;++t) {
            if (stats[MIN][i] > interval(t,i))
                stats[MIN][i] = interval(t,i);
            stats[MEAN][i] += interval(t,i);
            if (-stats[MAX][i] > -interval(t,i))
                stats[MAX][i] = interval(t,i);
        }
        stats[MEAN][i] /= (double)num_trials;
        stats[First][i] = interval(0,i);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    for (enum intervals i=start_interval;i<num_interval;++i) {
        OUTPUT(rank, i);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    for (int t=0;t<num_trials;++t) {
        DUMP_TIMES(rank, t);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    for (int t=0;t<num_trials;++t) {
        DUMP_INTERVALS(rank, t);
    }

    MPI_Finalize();
}
