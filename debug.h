#ifdef DEBUG

#include <stdio.h>

#define STDOUT_printf(RANK) printf("[Rank:%d] ", RANK);printf

#else

int DUMMY_printf ( const char * format, ... ) {return 0;}

#define STDOUT_printf(RANK) DUMMY_printf("[Rank:%d] ", RANK);DUMMY_printf

#endif

