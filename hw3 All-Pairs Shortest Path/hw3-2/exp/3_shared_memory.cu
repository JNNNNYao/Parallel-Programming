#include <stdio.h>
#include <stdlib.h>

// support different Blocking Factor (8, 16, 32, 64)
#define B_FACTOR 32

const int INF = ((1 << 30) - 1);

int V, E;
int pad_V;
int *Dist;

void input(char* infile) {
    FILE* file = fopen(infile, "rb");
    fread(&V, sizeof(int), 1, file);
    fread(&E, sizeof(int), 1, file);

    int remainder = V % 128;
    pad_V = (remainder == 0)? V: V + (128 - remainder);    // make V divisible by 128 for coalescing
    Dist = (int*)malloc(pad_V * pad_V * sizeof(int));

    for (int i = 0; i < pad_V; ++i) {
        for (int j = 0; j < pad_V; ++j) {
            if (i == j) {
                Dist[i * pad_V + j] = 0;
            } else {
                Dist[i * pad_V + j] = INF;
            }
        }
    }

    int pair[3];
    for (int i = 0; i < E; ++i) {
        fread(pair, sizeof(int), 3, file);
        Dist[pair[0] * pad_V + pair[1]] = pair[2];
    }
    fclose(file);
}

void output(char* outFileName) {
    FILE* outfile = fopen(outFileName, "w");
    for (int i = 0; i < V; ++i) {
        for (int j = 0; j < V; ++j) {
            if (Dist[i * pad_V + j] >= INF) Dist[i * V + j] = INF;  // remove padding
            else Dist[i * V + j] = Dist[i * pad_V + j]; // remove padding
        }
    }
    fwrite(Dist, sizeof(int), V*V, outfile);
    fclose(outfile);
}

__global__ void cal_phase1(int *D, int Round, int V) {
    __shared__ int shared[B_FACTOR][B_FACTOR];

    int pos_j = threadIdx.x;
    int pos_i = threadIdx.y * B_FACTOR/(blockDim.y);

    int block_round = Round * B_FACTOR;

    for (int i = 0; i < B_FACTOR/(blockDim.y); i++) {
        shared[(i + pos_i)][pos_j] = D[(block_round + (i + pos_i)) * V + (block_round + pos_j)];
    }
    __syncthreads();

    for (int k = 0; k < B_FACTOR; ++k) {
        for (int i = 0; i < B_FACTOR/(blockDim.y); i++) {
            if (shared[(i + pos_i)][pos_j] > shared[(i + pos_i)][k] + shared[k][pos_j])
                shared[(i + pos_i)][pos_j] = shared[(i + pos_i)][k] + shared[k][pos_j];
        }
        __syncthreads();
    }

    for (int i = 0; i < B_FACTOR/(blockDim.y); i++) {
        D[(block_round + (i + pos_i)) * V + (block_round + pos_j)] = shared[(i + pos_i)][pos_j];
    }
}

__global__ void cal_phase2(int *D, int Round, int V) {
    if (blockIdx.x == Round) {
        return;
    }

    __shared__ int pivot[B_FACTOR][B_FACTOR];
    __shared__ int row[B_FACTOR][B_FACTOR];
    __shared__ int col[B_FACTOR][B_FACTOR];

    int pos_j = threadIdx.x;
    int pos_i = threadIdx.y * B_FACTOR/(blockDim.y);

    int block = blockIdx.x * B_FACTOR;
    int block_round = Round * B_FACTOR;

    for (int i = 0; i < B_FACTOR/(blockDim.y); i++) {
        pivot[(i + pos_i)][pos_j] = D[(block_round + (i + pos_i)) * V + (block_round + pos_j)];
        // y-axis of block is Round
        // x-axis of block is blockIdx.x
        row[(i + pos_i)][pos_j] = D[(block_round + (i + pos_i)) * V + (block + pos_j)];
        // y-axis of block is blockIdx.x
        // x-axis of block is Round
        col[(i + pos_i)][pos_j] = D[(block + (i + pos_i)) * V + (block_round + pos_j)];
    }
    __syncthreads();

    for (int k = 0; k < B_FACTOR; ++k) {
        for (int i = 0; i < B_FACTOR/(blockDim.y); i++) {
            if (row[(i + pos_i)][pos_j] > pivot[(i + pos_i)][k] + row[k][pos_j])
                row[(i + pos_i)][pos_j] = pivot[(i + pos_i)][k] + row[k][pos_j];
            // be careful with the inequality
            if (col[(i + pos_i)][pos_j] > col[(i + pos_i)][k] + pivot[k][pos_j])
                col[(i + pos_i)][pos_j] = col[(i + pos_i)][k] + pivot[k][pos_j];
        }
        __syncthreads();
    }

    for (int i = 0; i < B_FACTOR/(blockDim.y); i++) {
        D[(block_round + (i + pos_i)) * V + (block + pos_j)] = row[(i + pos_i)][pos_j];
        D[(block + (i + pos_i)) * V + (block_round + pos_j)] = col[(i + pos_i)][pos_j];
    }
}

__global__ void cal_phase3(int *D, int Round, int V) {
    if (blockIdx.x == Round || blockIdx.y == Round) {
        return;
    }

    __shared__ int shared[B_FACTOR][B_FACTOR];
    __shared__ int row[B_FACTOR][B_FACTOR];
    __shared__ int col[B_FACTOR][B_FACTOR];

    int pos_j = threadIdx.x;
    int pos_i = threadIdx.y * B_FACTOR/(blockDim.y);

    int block_col = blockIdx.x * B_FACTOR;
    int block_row = blockIdx.y * B_FACTOR;
    int block_round = Round * B_FACTOR;

    for (int i = 0; i < B_FACTOR/(blockDim.y); i++) {
        // y-axis of block is row
        // x-axis of block is col
        shared[(i + pos_i)][pos_j] = D[(block_row + (i + pos_i)) * V + (block_col + pos_j)];
        // y-axis of block is row
        // x-axis of block is Round
        row[(i + pos_i)][pos_j] = D[(block_row + (i + pos_i)) * V + (block_round + pos_j)];
        // y-axis of block is Round
        // x-axis of block is col
        col[(i + pos_i)][pos_j] = D[(block_round + (i + pos_i)) * V + (block_col + pos_j)];
    }
    __syncthreads();

    for (int k = 0; k < B_FACTOR; ++k) {
        for (int i = 0; i < B_FACTOR/(blockDim.y); i++) {
            if (shared[(i + pos_i)][pos_j] > row[(i + pos_i)][k] + col[k][pos_j])
                shared[(i + pos_i)][pos_j] = row[(i + pos_i)][k] + col[k][pos_j];
        }
    }

    for (int i = 0; i < B_FACTOR/(blockDim.y); i++) {
        D[(block_row + (i + pos_i)) * V + (block_col + pos_j)] = shared[(i + pos_i)][pos_j];
    }
}

void block_FW(int B) {
    int *d_Dist;

    cudaHostRegister(Dist, pad_V * pad_V * sizeof(int), cudaHostRegisterDefault);
    cudaMalloc(&d_Dist, pad_V * pad_V * sizeof(int));
    cudaMemcpy(d_Dist, Dist, pad_V * pad_V * sizeof(int), cudaMemcpyHostToDevice);
    int y = (1024/B_FACTOR <= B_FACTOR)? 1024/B_FACTOR: B_FACTOR;
    dim3 num_threads(B_FACTOR, y);
    dim3 num_blocks(pad_V/B_FACTOR, pad_V/B_FACTOR);

    int round = pad_V / B;
    for (int r = 0; r < round; ++r) {
        // printf("%d %d\n", r, round);
        // fflush(stdout);
        /* Phase 1*/
        cal_phase1 <<<1, num_threads>>> (d_Dist, r, pad_V);

        /* Phase 2*/
        cal_phase2 <<<pad_V/B_FACTOR, num_threads>>> (d_Dist, r, pad_V);

        /* Phase 3*/
        cal_phase3 <<<num_blocks, num_threads>>> (d_Dist, r, pad_V);
    }
    cudaMemcpy(Dist, d_Dist, pad_V * pad_V * sizeof(int), cudaMemcpyDeviceToHost);
}

int main(int argc, char* argv[]) {
    input(argv[1]);
    block_FW(B_FACTOR);
    output(argv[2]);
    return 0;
}
