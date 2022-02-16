#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>

#define CHUNKSIZE 80

int main(int argc, char** argv) {
	if (argc != 3) {
		fprintf(stderr, "must provide exactly 2 arguments!\n");
		return 1;
	}
	MPI_Init(&argc, &argv);
    int mpi_rank, mpi_ranks;
	unsigned long long ans;
	MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_ranks);
	unsigned long long r = atoll(argv[1]);
	unsigned long long k = atoll(argv[2]);
	unsigned long long pixels = 0;
	int chunk = CHUNKSIZE;
	#pragma omp parallel for schedule(guided, chunk) reduction(+:pixels)
		for (unsigned long long x = mpi_rank; x < r; x = x+mpi_ranks) {
			unsigned long long y = ceil(sqrtl(r*r - x*x));
			pixels += y;
		}
	pixels %= k;
	MPI_Reduce(&pixels, &ans, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
	if (mpi_rank == 0) {
		printf("%llu\n", (4 * ans) % k);
	}
	MPI_Finalize();
}
