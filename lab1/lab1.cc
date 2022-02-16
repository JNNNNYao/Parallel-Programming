#include <assert.h>
#include <stdio.h>
#include <math.h>
#include <mpi.h>

int main(int argc, char** argv) {
	if (argc != 3) {
		fprintf(stderr, "must provide exactly 2 arguments!\n");
		return 1;
	}
	MPI_Init(&argc, &argv);
	int rank, npro;
	unsigned long long ans;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &npro);
	unsigned long long r = atoll(argv[1]);
	unsigned long long k = atoll(argv[2]);
	unsigned long long pixels = 0;
	for (unsigned long long x = rank; x < r; x = x+npro) {
		unsigned long long y = ceil(sqrtl(r*r - x*x));
		pixels += y;
		pixels %= k;
	}
	MPI_Reduce(&pixels, &ans, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
	if (rank == 0) {
		printf("%llu\n", (4 * ans) % k);
	}
	MPI_Finalize();
    return 0;
}
