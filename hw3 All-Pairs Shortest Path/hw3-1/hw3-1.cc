#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

const int INF = ((1 << 30) - 1);

int V, E;
static int Dist[6666][6666];

int num_threads;
pthread_barrier_t barr;

void input(char* infile) {
    FILE* file = fopen(infile, "rb");
    fread(&V, sizeof(int), 1, file);
    fread(&E, sizeof(int), 1, file);

    for (int i = 0; i < V; ++i) {
        for (int j = 0; j < V; ++j) {
            if (i == j) {
                Dist[i][j] = 0;
            } else {
                Dist[i][j] = INF;
            }
        }
    }

    int pair[3];
    for (int i = 0; i < E; ++i) {
        fread(pair, sizeof(int), 3, file);
        Dist[pair[0]][pair[1]] = pair[2];
    }
    fclose(file);
}

void output(char* outFileName) {
    FILE* outfile = fopen(outFileName, "w");
    for (int i = 0; i < V; ++i) {
        for (int j = 0; j < V; ++j) {
            if (Dist[i][j] >= INF) Dist[i][j] = INF;
        }
        fwrite(Dist[i], sizeof(int), V, outfile);
    }
    fclose(outfile);
}

int ceil(int a, int b) { return (a + b - 1) / b; }

void* task(void* arg_void_ptr) {
    int id = *(int*)arg_void_ptr;
	for (int k = 0; k < V; k++) {
        for (int i = id; i < V; i+=num_threads) {
            if (Dist[i][k] != INF && i != k) {
                for (int j = 0; j < V; j++) {
                    if (Dist[i][j] > Dist[i][k] + Dist[k][j]) {
                        Dist[i][j] = Dist[i][k] + Dist[k][j];
                    }
                }
            }
        }
        int res = pthread_barrier_wait(&barr);
    }
    pthread_exit(NULL);
}

int main(int argc, char* argv[]) {
    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    num_threads = CPU_COUNT(&cpu_set);

    // read input
    input(argv[1]);

    // calculation
    pthread_t threads[num_threads];
    pthread_barrier_init(&barr, NULL, num_threads);
	int args[num_threads];
    for(int i = 0; i < num_threads; i++){
		args[i] = i;
		pthread_create(&threads[i], NULL, task, (void*)&args[i]);
	}
	for(int i = 0; i < num_threads; i++){
		pthread_join(threads[i], NULL);
	}

    // write output
    output(argv[2]);
    return 0;
}
