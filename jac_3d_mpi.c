#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>

#define N (2*2*2*2*2*2+2)

#define A_FLOC(i,j,k) (A[(i)-startrow+1][j][k])
#define B_FLOC(i,j,k) (B[(i)-startrow][j][k])
// full locate:
// индексация эл-та, соотв. эл-ту в i-той строке
//   полной матрицы, в A или B

#define m_printf if(rank==0)printf

double maxeps = 0.1e-7;
int itmax = 100;
int i,j,k;
double eps;
double sum;
double A [N][N][N],  B [N][N][N];
// используются только части размеров (n_rows+2)*N*N и N*N*N в начале
//   матриц A и B соответственно
MPI_Request req[4];
MPI_Status status[4];
int startrow, lastrow, n_rows;
int n_ranks, rank;

void relax();
void resid();
void init();
void verify();

// для кол-ва процессов, равного 1, программа работать не будет
int main(int argc, char **argv)
{
    double t1, t2;
    int it;
    int code;
    if((code = MPI_Init(&argc, &argv)) != 0)
    {
        return code;
    }
    MPI_Comm_size(MPI_COMM_WORLD, &n_ranks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	t1 = MPI_Wtime();
	m_printf("N %d\n", N);
    m_printf("P %d\n", n_ranks);
    startrow = (rank * N) / n_ranks;
    lastrow = ((rank + 1) * N) / n_ranks - 1;
    n_rows = lastrow - startrow + 1;
    int stop = 0;
	init();
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Request req;
	for(it=1; it<=itmax; it++)
	{
		eps = 0.;
		relax();
		resid();
		m_printf("it=%4i   eps=%f\n", it, eps);
        if(eps < maxeps) {
            stop = 1;
        }
        MPI_Bcast(&stop, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if(stop) {
            break;
        }
	}
	verify();
	t2 = MPI_Wtime();
    printf("TIME %d: %lf\n", rank, t2 - t1);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
	return 0;
}

void init()
{
	for(i=startrow-1; i<=lastrow+1; i++)
	for(j=0; j<=N-1; j++)
	for(k=0; k<=N-1; k++)
	{
		if(i==0 || i==N-1 || j==0 || j==N-1 || k==0 || k==N-1)
			A_FLOC(i,j,k) = 0.;
		else
			A_FLOC(i,j,k) = 4.0 + i + j + k;
	}
}

void relax()
{
	for(i=startrow; i<=lastrow; i++) {
        if(i<=1 || i>=N-2)
            continue;
        for(j=1; j<=N-2; j++)
        for(k=1; k<=N-2; k++)
    	{
    		B_FLOC(i,j,k) = (
    			A_FLOC(i-1,j,k) + A_FLOC(i+1,j,k)
    			+ A_FLOC(i,j-1,k) + A_FLOC(i,j+1,k)
    			+ A_FLOC(i,j,k-1) + A_FLOC(i,j,k+1)
    		) / 6.;
    	}
    }
}

void resid()
{
	double e;
    double loceps = eps;
    int ll, shift;
    if(rank != 0)
        MPI_Irecv(&A_FLOC(startrow-1,0,0), N*N, MPI_DOUBLE, rank - 1, 1235,
            MPI_COMM_WORLD, &req[0]);
    if(rank != n_ranks - 1)
        MPI_Isend(&B_FLOC(lastrow,0,0), N*N, MPI_DOUBLE, rank + 1, 1235,
            MPI_COMM_WORLD, &req[2]);
    if(rank != n_ranks - 1)
        MPI_Irecv(&A_FLOC(lastrow+1,0,0), N*N, MPI_DOUBLE, rank + 1, 1236,
            MPI_COMM_WORLD, &req[3]);
    if(rank != 0)
        MPI_Isend(&B_FLOC(startrow,0,0), N*N, MPI_DOUBLE, rank - 1, 1236,
            MPI_COMM_WORLD, &req[1]);
    // здесь не происходит записи/чтения из 4 буферов,
    //   используемых выше для обмена
    for(i=startrow+1; i<=lastrow-1; i++)
	for(j=1; j<=N-2; j++)
	for(k=1; k<=N-2; k++)
	{
		e = fabs(A_FLOC(i,j,k) - B_FLOC(i,j,k));
		A_FLOC(i,j,k) = B_FLOC(i,j,k);
		if(e > loceps) loceps = e;
	}
    ll = 4;
    shift = 0;
    if(rank == 0) {
        ll = 2;
        shift = 2;
    }
    if(rank == n_ranks - 1) {
        ll = 2;
    }
    MPI_Waitall(ll, &req[shift], &status[0]);
    if(rank != n_ranks - 1) {
        for(j=1; j<=N-2; j++)
    	for(k=1; k<=N-2; k++)
    	{
            i = lastrow;
    		e = fabs(A_FLOC(i,j,k) - B_FLOC(i,j,k));
    		A_FLOC(i,j,k) = B_FLOC(i,j,k);
    		if(e > loceps) loceps = e;
    	}
    }
    if(rank != 0) {
        for(j=1; j<=N-2; j++)
    	for(k=1; k<=N-2; k++)
    	{
            i = startrow;
    		e = fabs(A_FLOC(i,j,k) - B_FLOC(i,j,k));
    		A_FLOC(i,j,k) = B_FLOC(i,j,k);
    		if(e > loceps) loceps = e;
    	}
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Reduce(&loceps, &eps, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
}

void verify()
{
	double s = 0.;
	for(i=startrow; i<=lastrow; i++)
    for(j=0; j<=N-1; j++)
    for(k=0; k<=N-1; k++)
    {
		s += A_FLOC(i,j,k)*(i+1)*(j+1)*(k+1)/(N*N*N);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Reduce(&s, &sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	m_printf("  S = %f\n", sum);
}
