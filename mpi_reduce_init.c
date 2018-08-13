#include <mpi.h>
#include "mpi_reduce_init.h"
#include "mpi_try_passive.h"

#include "debug.h"

// INTENT: BUILD MPI_REDUCE OUT OF EXISTING RMA FUNCTIONS
// PLAN  : INIT - create window(s) and determine schedule - MPI_WIN_ATTACH(&recvbuf, &win); if(root) MPI_WIN_LOCK(EXCLUSIVE)
//         START - begin to execute schedule of RMA calls - MPI_WIN_ACCUMULATE(sendbuf, op, &win)
//         TEST - continue to execute remaining RMA calls - MPI_WIN_UNLOCK(root,win); if(root) MPI_WIN_LOCK(EXCLUSIVE)
//         FREE - destroy the window(s) and free schedule - MPI_WIN_FREE(&win)
// IMAGE : https://www.websequencediagrams.com/cgi-bin/cdraw?lz=dGl0bGUgUGVyc2lzdGVudCBjb2xsZWN0aXZlIHVzaW5nIFJNQQoKbm90ZSBvdmVyICBQUk9DXzAsIFJPT1QsIFdJTl8xAAEGMiwgT1RIRVIsAB0GTgpJbml0aWFsaXppbmcKZW5kIG5vdGUKCnBhcnRpY2lwYW50AEYHAAYNUk9PVAAXDQBeBQABETIAOw0AdAUASBJOCgoAgSkGLT4rUk9PVDogTVBJLTxwY29sbD4tSU5JVAAaBk4tPisAgTcFAA4TUk9PVC0-KgCBZAU6IENyZWF0ZQCBDQcAgWoFLT4ABxQAKwsyACkNMgAqDAALEABmBwBlB0F0dGFjaCByZWN2YnVmIHRvAE8NAFQHAA8WMgCBMAcrAIEwB0xvY2sgZXhjbHVzaXYAgRYOAB4FMgAOFQCBCgc8LT4AghYHQkFSUklFUgCCDgYtPi0AhAYGAIJRBV9TVUNDRVNTAIIOBwAVCE4ADw4AhBM2U3RhcnRpbmcgKG9kZCkAhEALAINCG1NUQVIAg0AeABwGAIVQCwCEPwZjYWxsAIQaDVRFU1QAhTUKYWx0IHRvbyBlYXJseQCBZAkAhBcHZmFpbCBsb2NrAIUuB2VuZACBcR8Ag2QNUFVUIHNlbmRidWYgaW50bwCDcwgKYWN0aXYAhE0QAIULB1VuAGwFAINNEGRlACYPAAEQAIVSBwCGEQZJYmFycmllcgCDPR0Ah1A2VGVzAIMnFWxvb3AgdW50aWwgZmxhZyBpcyAxIHsAhnMdAIMFBQCCfQVzdGlsbACCXStsc2UgZXhwb3NlAIMJCACGEglUcnkAgy0Gc2hhcmVkAIdGFUFjY3VtdWxhdGUAgxIOdGFyZ2V0AFkGY29udGVudGlvbgCEBhBUcnkgdQCDIAYAVw19AIESBmF2YWlsYWJsZQCIQQ8AHxgAg0kRAIkABwCJOgcAg0IJAIFwBQByBgCBbAoAiV8HdGVzdCBpAINtCH0AhQciAB0GAIMgFQCKRxwAhjcFAIRdDABrDgCIfSgAiFQXAIEOBSMjIyMjIwABBQABCgAPBgCILkFldmVuAIdXgRkyAIhgEDIAiEMtMgCISCgAjEQIAI1FBwCIWxUyAIhgEAABEACNCQkAiAZqAIMtEACIGFkAgnUUAIhZGDIAiFsWAJAGEACINjsAWwcAiF8SMgCIVB0AHBsAg0oRAIcsgUUAkVsYAIhQIQCVVzVGcmVlAJV5DgCUfhtGUkVFAJR8HQAcBQCPchQAj04dAJU8B0RlAJRKDWZyb20AlEUUAEcLAIc0GQCVWAcAOhgAlQAGAJQtEgCWKg5GcmUAlj4TMgAOCwCWLAgAlD4XAJYoDQBDC2Rlc3Ryb3kAliIUAFALABoMAJZ6CACRaRgK&s=napkin

const unsigned int false = 0, true = 1;

const MPIX_Request *const MPIX_REQUEST_NULL = 0;

int MPIX_REDUCE_INIT(const void* sendbuf, void* recvbuf,
                     int count, MPI_Datatype datatype,
                     MPI_Op op, int root,
                     MPI_Comm comm, MPI_Info info,
                     MPIX_Request *request) {

    int retVal = MPI_SUCCESS;

    request->isactive = false;
    request->isinplace = MPI_IN_PLACE==sendbuf ? true : false;
    request->isstarting = false;
    request->toggle = 1;
    request->comm = comm;
    MPI_Comm_rank(comm, &request->myrank);

STDOUT_printf(request->myrank)
      ("Input parameters, sendbuf=%p (value=%li), recvbuf=%p (value=%li), count=%d\n",
       sendbuf, *(long*)sendbuf,
       recvbuf, *(long*)recvbuf,
       count);

    if (op == MPI_SUM ||
        op == MPI_PROD) {
        // any built-in operator that is supported by MPI RMA and RDMA hardware
        // tactics: create a window from recvbuf, then use locks and accumulate

        // step 0; create a dynamic window x2

        MPI_Count true_lb = 0, true_extent = 0;
        MPI_Type_get_true_extent_x(datatype, &true_lb, &true_extent);
        MPI_Win_create_dynamic(MPI_INFO_NULL, comm, &request->win[0]);
        MPI_Win_create_dynamic(MPI_INFO_NULL, comm, &request->win[1]);

        MPI_Aint base = 0;
        if(request->myrank==root) {
            request->iamroot = true;

            // step 1: attach the user recvbuf x2

STDOUT_printf(request->myrank)
      ("About to calculate base address for win\n");
            MPI_Get_address(recvbuf, &base);
STDOUT_printf(request->myrank)
      ("Converted recvbuf address (%p) to MPI_AINT location (%li) which looks like address (%p)\n",
       recvbuf, base, (void*)base);
            MPI_Aint_add(base, (MPI_Aint)true_lb);
STDOUT_printf(request->myrank)
      ("Adjusted base by %lli to %p\n",
       true_lb, (void*)base);
STDOUT_printf(request->myrank)
      ("About to attach memory to 1st win, using base=%p, extent=%lli\n",
       (void*)base, true_extent);
            MPI_Win_attach(request->win[0], (void*)base, true_extent);
STDOUT_printf(request->myrank)
      ("About to attach memory to 2nd win, using base=%p, extent=%lli\n",
       (void*)base, true_extent);
            MPI_Win_attach(request->win[1], (void*)base, true_extent);
STDOUT_printf(request->myrank)
      ("Done attaching memory to both win\n");

            // step 2: lock window at the root x2

            // important: the root issues an exclusive lock during initialization
            //            to enforce correct synchronization when starting the op
            //            Remote calls to MPI_START must not affect local memory,
            //            until the local process permits it by calling MPI_START
            //            This lock protects local load/store instructions issued
            //            between initialization and the first call to MPI_START,
            //            or the call to MPI_REQUEST_FREE, whichever comes first.

            int assert = 0; // must not give MPI_MODE_NOCHECK here - other ranks *will* issue conflicting locks
            MPI_Win_lock(MPI_LOCK_EXCLUSIVE, root, assert, request->win[0]);
            MPI_Win_lock(MPI_LOCK_EXCLUSIVE, root, assert, request->win[1]);
        }

        // step 3: preclude pre-attach access
        MPI_Bcast(&base, 1, MPI_AINT, root, comm);

        // step 4: record schedule parameters
        request->sched.origin_addr     = sendbuf;
        request->sched.origin_count    = count;
        request->sched.origin_datatype = datatype;
        request->sched.target_rank     = root;
        request->sched.target_disp     = base;
        request->sched.target_count    = count;
        request->sched.target_datatype = datatype;
        request->sched.op              = op;

    } else {
        //MPIX_Reduce_init(sendbuf, recvbuf, count, datatype, op, root, comm, info, request);
        retVal = MPI_ERR_INTERN;
    }

    return retVal;
}

int MPIX_REDUCE_START(MPIX_Request *const request) {
    int retVal = MPI_SUCCESS;

    request->toggle = 1 - request->toggle;
    request->isstarting = true;
    request->isactive = true;

    MPI_Win *win = &request->win[request->toggle];

    if (request->iamroot) {
        // root participant

        if(!request->isinplace) {
            // copy my sendbuf into my recvbuf with put operation
            // before remote ranks do their accumulate operations
STDOUT_printf(request->myrank)
      ("About to Put from (addr=%p,count=%lli) to rank %d at (disp=%li,count=%lli)\n",
       request->sched.origin_addr,
       request->sched.origin_count,
       request->sched.target_rank,
       request->sched.target_disp,
       request->sched.target_count);
            MPI_Put(request->sched.origin_addr,
                    request->sched.origin_count, request->sched.origin_datatype,
                    request->sched.target_rank, request->sched.target_disp,
                    request->sched.target_count, request->sched.target_datatype,
                  //request->sched.op,
                    *win);
STDOUT_printf(request->myrank)
      ("Done Put\n");
        } // else { /* is in place, user has already filled recvbuf for us */ }

        // release the local MPI_MODE_EXCLUSIVE lock, temporarily
        // (first taken during initialization, temporarily released when starting, and retaken during completion)
STDOUT_printf(request->myrank)
      ("About to unlock\n");
        MPI_Win_unlock(request->myrank, *win);
STDOUT_printf(request->myrank)
      ("Done unlock\n");

        // notify that this process has contributed its data
        request->state = isUnlocked;

    } else {
        // non-root participant
        request->state = isEarly;
    }

    // make progress by beginning execution of the schedule
    int flag = 0; MPI_Status status;
    retVal = MPIX_REDUCE_REQUEST_GET_STATUS(request, &flag, &status);

    return retVal;
}

int MPIX_REDUCE_TEST(MPIX_Request *const request, int *const flag, MPI_Status *const status) {
    int retVal = MPI_SUCCESS;

    MPIX_REDUCE_REQUEST_GET_STATUS(request, flag, status);
    if(MPI_SUCCESS==retVal && *flag) {
        retVal = MPI_Test(&request->req_ib, flag, status);
        if(MPI_SUCCESS==retVal && *flag) {
            request->isactive = false;
        }
    }

    return retVal;
}

int MPIX_REDUCE_REQUEST_GET_STATUS(MPIX_Request *const request, int *const flag, MPI_Status *const status) {
    int retVal = MPI_SUCCESS;

    MPI_Win *win = &request->win[request->toggle];
    int root = request->sched.target_rank;

    switch(request->state) {

      case isEarly: {
        int locked = 0, assert = 0;
STDOUT_printf(request->myrank)
      ("About to try lock\n");
        retVal = MPIX_WIN_TRYLOCK(MPI_LOCK_SHARED, root, assert, *win, &locked);
        if(MPI_SUCCESS==retVal && locked) {
STDOUT_printf(request->myrank)
      ("Done try lock OK\n");
            request->state = isExposed;
        } else {
STDOUT_printf(request->myrank)
      ("Done try lock BAD\n");
            break;
        }
      }

      case isExposed:
STDOUT_printf(request->myrank)
      ("About to accumulate from (addr=%p,count=%lli,value=%li) to rank %d at (disp=%li,count=%lli)\n",
       request->sched.origin_addr,
       request->sched.origin_count,
       *(long*)request->sched.origin_addr,
       request->sched.target_rank,
       request->sched.target_disp,
       request->sched.target_count);
        retVal = MPI_Accumulate(request->sched.origin_addr,
                                request->sched.origin_count, request->sched.origin_datatype,
                                request->sched.target_rank, request->sched.target_disp,
                                request->sched.target_count, request->sched.target_datatype,
                                request->sched.op,
                                *win);
        if(MPI_SUCCESS==retVal) {
STDOUT_printf(request->myrank)
      ("Done accumulate OK\n");
            request->state = isContention;
        } else {
STDOUT_printf(request->myrank)
      ("Done accumulate BAD\n");
            break;
        }

      case isContention: {
        int unlocked = 0;
STDOUT_printf(request->myrank)
      ("About to try unlock\n");
        retVal = MPIX_WIN_TRYUNLOCK(root, *win, &unlocked);
        if(MPI_SUCCESS==retVal && unlocked) {
STDOUT_printf(request->myrank)
      ("Done try unlock OK\n");
            request->state = isUnlocked;
        } else {
STDOUT_printf(request->myrank)
      ("Done try unlock BAD\n");
            break;
        }
      }

      case isUnlocked:
STDOUT_printf(request->myrank)
      ("About to ibarrier\n");
        retVal = MPI_Ibarrier(request->comm, &request->req_ib);
        if(MPI_SUCCESS==retVal) {
STDOUT_printf(request->myrank)
      ("Done ibarrier OK\n");
            request->state = isBlocked;
        } else {
STDOUT_printf(request->myrank)
      ("Done ibarrier BAD\n");
            break;
        }

      case isBlocked:
STDOUT_printf(request->myrank)
      ("About to get status of ibarrier\n");
        retVal = MPI_Request_get_status(request->req_ib, flag, status);
        if(MPI_SUCCESS==retVal && *flag) {
STDOUT_printf(request->myrank)
      ("Done get status of ibarrier OK\n");
            request->state = isComplete;
            if(request->iamroot) {
                int assert = 0; // must not give MPI_MODE_NOCHECK here - other ranks *will* issue conflicting locks
STDOUT_printf(request->myrank)
      ("About to lock\n");
                MPI_Win_lock(MPI_LOCK_EXCLUSIVE, root, assert, *win);
STDOUT_printf(request->myrank)
      ("Done lock\n");
            }
        } else {
STDOUT_printf(request->myrank)
      ("Done get status of ibarrier BAD\n");
            break;
        }

      case isComplete:
        *flag = true;
        break;

      default:
        break;
    }

    return retVal;
}

int MPIX_REDUCE_FREE(MPIX_Request *request) {
    int retVal = MPI_SUCCESS;

STDOUT_printf(request->myrank)
      ("In request free\n");
    if(request->iamroot) {
        // release the local MPI_MODE_EXCLUSIVE lock, for the last time
        // (first taken during initialization, temporarily released when starting, and retaken during completion)
        MPI_Win_unlock(request->myrank, request->win[0]);
        MPI_Win_unlock(request->myrank, request->win[1]);
    }

    // note that free implicitly detaches all attached memory
    MPI_Win_free(&request->win[0]);
    MPI_Win_free(&request->win[1]);

    request = (MPIX_Request*)MPIX_REQUEST_NULL;

    return retVal;
}

