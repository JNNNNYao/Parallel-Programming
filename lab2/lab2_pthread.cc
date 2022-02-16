#include <pthread.h>
#include <assert.h>
#include <stdio.h>
#include <math.h>

unsigned long long r, k, ans;
int num_threads;

struct argument{
	int ID;
	unsigned long long pixels;
};

void* Loop(void* arg_void_ptr){
    argument* arg = static_cast<argument*>(arg_void_ptr);
	unsigned long long pixels = 0;
    for(unsigned long long x = arg->ID; x < r; x = x+num_threads){
		unsigned long long y = ceil(sqrtl(r*r - x*x));
		pixels += y;
	}
	arg->pixels += pixels;
    pthread_exit(NULL);
}

int main(int argc, char** argv) {
	if (argc != 3) {
		fprintf(stderr, "must provide exactly 2 arguments!\n");
		return 1;
	}
	r = atoll(argv[1]);
	k = atoll(argv[2]);
	ans = 0;
	cpu_set_t cpuset;
	sched_getaffinity(0, sizeof(cpuset), &cpuset);
	unsigned long long ncpus = CPU_COUNT(&cpuset);

	num_threads = ncpus;
	pthread_t threads[num_threads];
	argument args[num_threads];

	for(int i = 0; i < num_threads; i++){
		args[i].ID = i;
		args[i].pixels = 0;
		pthread_create(&threads[i], NULL, Loop, (void*)&args[i]);
	}
	for(int i = 0; i < num_threads; i++){
		pthread_join(threads[i], NULL);
		ans += args[i].pixels;
	}

	ans %= k;

	printf("%llu\n", (4 * ans) % k);
	pthread_exit(NULL);
}