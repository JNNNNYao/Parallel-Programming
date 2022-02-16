#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// 49152(shared memory in bytes)/4(size of int)/3(phase2 & phase3 need three matrix) = 4096 = 64*64
#define B_FACTOR 64

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
    int pos_i = threadIdx.y * 4;

    int block_round = Round * B_FACTOR;

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        shared[(i + pos_i)][pos_j] = D[(block_round + (i + pos_i)) * V + (block_round + pos_j)];
    }
    __syncthreads();

    for (int k = 0; k < B_FACTOR; ++k) {
        shared[pos_i][pos_j] = min(
            shared[pos_i][pos_j],
            shared[pos_i][k] + shared[k][pos_j]
        );
        shared[(1 + pos_i)][pos_j] = min(
            shared[(1 + pos_i)][pos_j],
            shared[(1 + pos_i)][k] + shared[k][pos_j]
        );
        shared[(2 + pos_i)][pos_j] = min(
            shared[(2 + pos_i)][pos_j],
            shared[(2 + pos_i)][k] + shared[k][pos_j]
        );
        shared[(3 + pos_i)][pos_j] = min(
            shared[(3 + pos_i)][pos_j],
            shared[(3 + pos_i)][k] + shared[k][pos_j]
        );
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < 4; i++) {
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
    int pos_i = threadIdx.y * 4;

    int block = blockIdx.x * B_FACTOR;
    int block_round = Round * B_FACTOR;

    #pragma unroll
    for (int i = 0; i < 4; i++) {
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
        row[pos_i][pos_j] = min(
            row[pos_i][pos_j],
            pivot[pos_i][k] + row[k][pos_j]
        );
        row[(1 + pos_i)][pos_j] = min(
            row[(1 + pos_i)][pos_j],
            pivot[(1 + pos_i)][k] + row[k][pos_j]
        );
        row[(2 + pos_i)][pos_j] = min(
            row[(2 + pos_i)][pos_j],
            pivot[(2 + pos_i)][k] + row[k][pos_j]
        );
        row[(3 + pos_i)][pos_j] = min(
            row[(3 + pos_i)][pos_j],
            pivot[(3 + pos_i)][k] + row[k][pos_j]
        );
        // be careful with the inequality
        col[pos_i][pos_j] = min(
            col[pos_i][pos_j],
            col[pos_i][k] + pivot[k][pos_j]
        );
        col[(1 + pos_i)][pos_j] = min(
            col[(1 + pos_i)][pos_j],
            col[(1 + pos_i)][k] + pivot[k][pos_j]
        );
        col[(2 + pos_i)][pos_j] = min(
            col[(2 + pos_i)][pos_j],
            col[(2 + pos_i)][k] + pivot[k][pos_j]
        );
        col[(3 + pos_i)][pos_j] = min(
            col[(3 + pos_i)][pos_j],
            col[(3 + pos_i)][k] + pivot[k][pos_j]
        );
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        D[(block_round + (i + pos_i)) * V + (block + pos_j)] = row[(i + pos_i)][pos_j];
        D[(block + (i + pos_i)) * V + (block_round + pos_j)] = col[(i + pos_i)][pos_j];
    }
}

__global__ void cal_phase3(int *D, int Round, int V) {
    if (blockIdx.x == Round || blockIdx.y == Round) {
        return;
    }

    __shared__ int row[B_FACTOR][B_FACTOR];
    __shared__ int col[B_FACTOR][B_FACTOR];

    int pos_j = threadIdx.x;
    int pos_i = threadIdx.y * 4;

    int block_col = blockIdx.x * B_FACTOR;
    int block_row = blockIdx.y * B_FACTOR;
    int block_round = Round * B_FACTOR;

    // y-axis of block is row
    // x-axis of block is col
    int i_j_0 = D[(block_row + pos_i) * V + (block_col + pos_j)];
    int i_j_1 = D[(block_row + (1 + pos_i)) * V + (block_col + pos_j)];
    int i_j_2 = D[(block_row + (2 + pos_i)) * V + (block_col + pos_j)];
    int i_j_3 = D[(block_row + (3 + pos_i)) * V + (block_col + pos_j)];
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        // y-axis of block is row
        // x-axis of block is Round
        row[(i + pos_i)][pos_j] = D[(block_row + (i + pos_i)) * V + (block_round + pos_j)];
        // y-axis of block is Round
        // x-axis of block is col
        col[(i + pos_i)][pos_j] = D[(block_round + (i + pos_i)) * V + (block_col + pos_j)];
    }
    __syncthreads();

    for (int k = 0; k < B_FACTOR; ++k) {
        i_j_0 = min(i_j_0, row[(pos_i)][k] + col[k][pos_j]);
        i_j_1 = min(i_j_1, row[(1 + pos_i)][k] + col[k][pos_j]);
        i_j_2 = min(i_j_2, row[(2 + pos_i)][k] + col[k][pos_j]);
        i_j_3 = min(i_j_3, row[(3 + pos_i)][k] + col[k][pos_j]);
    }

    D[(block_row + pos_i) * V + (block_col + pos_j)] = i_j_0;
    D[(block_row + (1 + pos_i)) * V + (block_col + pos_j)] = i_j_1;
    D[(block_row + (2 + pos_i)) * V + (block_col + pos_j)] = i_j_2;
    D[(block_row + (3 + pos_i)) * V + (block_col + pos_j)] = i_j_3;
}

void block_FW(int B) {
    int *d_Dist;

    cudaHostRegister(Dist, pad_V * pad_V * sizeof(int), cudaHostRegisterDefault);
    cudaMalloc(&d_Dist, pad_V * pad_V * sizeof(int));
    cudaMemcpy(d_Dist, Dist, pad_V * pad_V * sizeof(int), cudaMemcpyHostToDevice);
    dim3 num_threads(B_FACTOR, 1024/B_FACTOR);  // 64, 16
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
