DIM ?= 16
KNN ?= 32
TRA ?= 1048576
QUE ?= 1024

LOW ?=  0
HIGH ?= 2


CFLAGS  = -DPROBDIM=$(DIM) -DNNBS=$(KNN) -DTRAINELEMS=$(TRA) -DQUERYELEMS=$(QUE) -DLB=$(LOW) -DUB=$(HIGH) -O3 #-g -ggdb 
CFLAGS += -DSURROGATES -Wall -fopenmp
LDFLAGS += -lm -fopenmp

all: gendata myknn myknn_tasks

gendata: gendata.o
	gcc -o gendata gendata.o $(LDFLAGS)

gendata.o: gendata.c func.c
	gcc $(CFLAGS) -c gendata.c

myknn: myknn.o
	gcc -o myknn myknn.o $(LDFLAGS)

myknn.o: myknn.c func.c
	gcc $(CFLAGS) -c myknn.c

myknn_tasks: myknn_tasks.o
	gcc -o myknn_tasks myknn_tasks.o $(LDFLAGS)

myknn_tasks.o: myknn_tasks.c func.c
	gcc $(CFLAGS) -c myknn_tasks.c

clean:
	rm -f myknn myknn_tasks *.o gendata

	
