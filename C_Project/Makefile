DIM ?= 16
KNN ?= 32
TRA ?= 1048576
QUE ?= 1024

LOW ?=  0
HIGH ?= 2


CFLAGS  = -DPROBDIM=$(DIM) -DNNBS=$(KNN) -DTRAINELEMS=$(TRA) -DQUERYELEMS=$(QUE) -DLB=$(LOW) -DUB=$(HIGH) -O3 #-g -ggdb 
CFLAGS += -DSURROGATES -Wall -fopenmp
LDFLAGS += -lm -fopenmp -pg

all: gendata myknn myknn_parallel myknn_tasks myknn_mpi

gendata: gendata.o
	gcc -o gendata gendata.o $(LDFLAGS)

gendata.o: gendata.c func.c
	gcc $(CFLAGS) -c gendata.c

myknn: myknn.o
	gcc -o myknn myknn.o $(LDFLAGS)

myknn.o: myknn.c func.c
	gcc $(CFLAGS) -c myknn.c

myknn_parallel: myknn_parallel.o
	gcc -o myknn_parallel myknn_parallel.o $(LDFLAGS)

myknn_parallel.o: myknn_parallel.c func.c
	gcc $(CFLAGS) -c myknn_parallel.c

myknn_tasks: myknn_tasks.o
	gcc -o myknn_tasks myknn_tasks.o $(LDFLAGS)

myknn_tasks.o: myknn_tasks.c func.c
	gcc $(CFLAGS) -c myknn_tasks.c

myknn_mpi: myknn_mpi.o
	mpicc -o myknn_mpi myknn_mpi.o $(LDFLAGS)

myknn_mpi.o: myknn_mpi.c func.c
	mpicc $(CFLAGS) -c myknn_mpi.c

clean:
	rm -f myknn myknn_parallel myknn_tasks myknn_mpi *.o gendata

	
