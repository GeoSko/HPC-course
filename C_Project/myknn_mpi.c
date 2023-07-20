#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <mpi.h>


#ifndef PROBDIM
#define PROBDIM 2
#endif

#include "func.c"

static double **xdata;
static double ydata[TRAINELEMS];

static double **xtest;
static double ytest[QUERYELEMS];

#define MAX_NNB	256

double find_knn_value(double *p, int n, int knn)
{
	int nn_x[MAX_NNB];
	double nn_d[MAX_NNB];

	compute_knn_brute_force(xdata, p, TRAINELEMS, PROBDIM, knn, nn_x, nn_d); // brute-force /linear search

	int dim = PROBDIM;
	int nd = knn;   // number of points
	double xd[MAX_NNB*PROBDIM];   // points
	double fd[MAX_NNB];     // function values

	for (int i = 0; i < knn; i++) {
		fd[i] = ydata[nn_x[i]];
	}

	for (int i = 0; i < knn; i++) {
		for (int j = 0; j < PROBDIM; j++) {
			xd[i*dim+j] = xdata[nn_x[i]][j];
		}
	}

	double fi;

	fi = predict_value(dim, nd, xd, fd, p, nn_d);

	return fi;
}

int main(int argc, char *argv[])
{
	MPI_Init(&argc, &argv);
	int rank;
	int size;
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	MPI_Comm_size(MPI_COMM_WORLD,&size);
	

	if (argc != 3)
	{
		printf("usage: %s <trainfile> <queryfile>\n", argv[0]);
		exit(1);
	}

	char *trainfile = argv[1];
	char *queryfile = argv[2];

	double *xmem = (double *)malloc(TRAINELEMS*PROBDIM*sizeof(double));
	xdata = (double **)malloc(TRAINELEMS*sizeof(double *));
	// #pragma omp paralel for
	for (int i = 0; i < TRAINELEMS; i++) xdata[i] = xmem + i*PROBDIM; //&mem[i*PROBDIM];

	FILE *fpin = open_traindata(trainfile);

	for (int i=0;i<TRAINELEMS;i++) {
		for (int k = 0; k < PROBDIM; k++)
			xdata[i][k] = read_nextnum(fpin);
#if defined(SURROGATES)
			ydata[i] = read_nextnum(fpin);
#else
			ydata[i] = 0;
#endif
	}
	fclose(fpin);

	fpin = open_querydata(queryfile);

	//FILE *fpout = fopen("output.knn.txt","w");

	double *y = malloc(QUERYELEMS*sizeof(double));

	// double t0, t1;
	double t_first = 0.0, t_sum_local = 0.0, t_sum_global = 0.0;
	double local_sse = 0.0;
	double global_sse = 0.0;
	double err, global_err_sum = 0.0, local_err_sum = 0.0;
	double x[PROBDIM];

	double *xmem_test = (double *)malloc(QUERYELEMS*PROBDIM*sizeof(double));
	xtest = (double **)malloc(QUERYELEMS*sizeof(double *));

	for (int i = 0; i < QUERYELEMS; i++) xtest[i] = xmem_test + i*PROBDIM; //&mem[i*PROBDIM];

	for (int i=0;i<QUERYELEMS;i++) {
			for (int k = 0; k < PROBDIM; k++)
				xtest[i][k] = read_nextnum(fpin);
	#if defined(SURROGATES)
				ytest[i] = read_nextnum(fpin);
	#else
				ytest[i] = 0;
	#endif
	}
	fclose(fpin);


// 	for (int i=0;i<QUERYELEMS;i++) {	/* requests */
// 		for (int k = 0; k < PROBDIM; k++)
// 			x[k] = read_nextnum(fpin);
// #if defined(SURROGATES)
// 		y[i] = read_nextnum(fpin);
// #else
// 		y[i] = 0.0;
// #endif

	int batch = QUERYELEMS/size;
	int start = rank*batch;
	int end = start+batch;

	// printf("rank: %d\nqueryelems: %d\nbatch: %d\nstart: %d\nend:%d\n\n",rank, QUERYELEMS, batch, start, end);



	// t0 = omp_get_wtime();
	double t1 = MPI_Wtime();
	for (int i=start;i<end;i++) {
		// t0 = gettime();
		
		double yp = find_knn_value(xtest[i], PROBDIM, NNBS);
		// t1 = gettime();

		local_sse += (ytest[i]-yp)*(ytest[i]-yp);

		//for (k = 0; k < PROBDIM; k++)
		//	fprintf(fpout,"%.5f ", x[k]);

		err = 100.0*fabs((yp-ytest[i])/ytest[i]);
		//fprintf(fpout,"%.5f %.5f %.2f\n", y[i], yp, err);
		local_err_sum += err;

	}
	// t1 = omp_get_wtime();
	double t2 = MPI_Wtime();
	double t_local = t2-t1;
	// t_sum_local = (t1-t0);
	// fclose(fpin);
	//fclose(fpout);

	MPI_Reduce(&local_sse, &global_sse, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	MPI_Reduce(&local_err_sum, &global_err_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	// MPI_Reduce(&t_sum_local, &t_sum_local, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);


	if (rank==0){ // only one prints
		printf("rank %d:\n", rank);
		double mse = global_sse/QUERYELEMS;
		
		double ymean = compute_mean(ytest, QUERYELEMS);
		double var = compute_var(ytest, QUERYELEMS, ymean);
		double r2 = 1-(mse/var);

		printf("Results for %d query points\n", QUERYELEMS);
		printf("APE = %.2f %%\n", global_err_sum/QUERYELEMS);
		printf("MSE = %.6f\n", mse);
		printf("R2 = 1 - (MSE/Var) = %.6lf\n", r2);

		t_local = t_local*1000.0;			// convert to ms
		// t_first = t_first*1000.0;	// convert to ms
		printf("Total time = %lf ms\n", t_local);
		// printf("Time for 1st query = %lf ms\n", t_first);
		// printf("Time for 2..N queries = %lf ms\n", t_sum-t_first);
		printf("Average time/query = %lf ms\n", (t_local)/(QUERYELEMS));
	} 

	
	MPI_Finalize();
	return 0;
}
