#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#define  Max(a,b) ((a)>(b)?(a):(b))

#define N (2*2*2*2*2*2+2)
double maxeps = 0.1e-7;
int itmax = 100;
int i,j,k;
double eps;
double A [N][N][N],  B [N][N][N];

int n_threads = 1;

void relax();
void resid();
void init();
void verify();

int main(int argc, char **argv)
{
	if(argc > 1) {
		sscanf(argv[1], "%d", &n_threads);
	}
    int iter;
	omp_set_dynamic(0);
	omp_set_num_threads(n_threads);
	double start_time = omp_get_wtime();
	int it;
	init();
	for(it=1; it<=itmax; it++)
	{
		eps = 0.;
		relax();
		resid();
		printf( "it=%4i   eps=%f\n", it, eps);
		if (eps < maxeps) break;
	}
	verify();
	double time = omp_get_wtime() - start_time;
	printf("TIME %g\n", time);
	return 0;
}

void init()
{
	#pragma omp parallel for private(i,j,k) shared(A)
	for(i=0; i<=N-1; i++)
	for(j=0; j<=N-1; j++)
	for(k=0; k<=N-1; k++)
	{
		if(i==0 || i==N-1 || j==0 || j==N-1 || k==0 || k==N-1)
			A[i][j][k]= 0.;
		else
			A[i][j][k] = ( 4.0 + i + j + k);
	}
}

void relax()
{
	#pragma omp parallel for private(i,j,k) shared(A, B)
	for(i=1; i<=N-2; i++)
	for(j=1; j<=N-2; j++)
	for(k=1; k<=N-2; k++)
	{
		B[i][j][k] = (
			A[i-1][j][k] + A[i+1][j][k]
			+ A[i][j-1][k] + A[i][j+1][k]
			+ A[i][j][k-1] + A[i][j][k+1]
		) / 6.;
	}
}

void resid()
{
	double e;
	#pragma omp parallel for private(i,j,k,e) shared(A, B, eps) reduction(max: maxeps)
	for(i=1; i<=N-2; i++)
	for(j=1; j<=N-2; j++)
	for(k=1; k<=N-2; k++)
	{
		e = fabs(A[i][j][k] - B[i][j][k]);
		A[i][j][k] = B[i][j][k];
		if(e > eps) eps = e;
	}
}

void verify()
{
	double s = 0.;
	#pragma omp parallel for private(i,j,k) shared(A) reduction(+:s)
	for(i=0; i<=N-1; i++)
	for(j=0; j<=N-1; j++)
	for(k=0; k<=N-1; k++)
	{
		s+=A[i][j][k]*(i+1)*(j+1)*(k+1)/(N*N*N);
	}
	printf("  S = %f\n",s);
}
