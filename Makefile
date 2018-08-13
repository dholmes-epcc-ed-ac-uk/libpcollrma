
CC = mpicc
#CFLAGS = -DDEBUG

PROGRAM = main_test
SOURCES = mpi_try_passive.c \
	mpi_reduce_init.c
OBJECTS := $(SOURCES:.c=.o)

.PHONY: all clean

all: $(PROGRAM)

clean:
	rm -f $(PROGRAM) $(OBJECTS)

$(PROGRAM): $(OBJECTS) $(addsuffix .c,$(PROGRAM)) *.h
	$(CC) $(addsuffix .c,$(PROGRAM)) -o $(PROGRAM) $(OBJECTS)
