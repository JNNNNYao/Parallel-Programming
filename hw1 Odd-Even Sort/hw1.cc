#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
// #include <boost/sort/spreadsort/float_sort.hpp>

// using namespace boost::sort::spreadsort;

// struct rightshift{
// inline int operator()(const float &x, const unsigned offset) const {
//     return float_mem_cast<float, int>(x) >> offset;
//   }
// };

inline unsigned int FloatToRadix(float value)
{
	unsigned int radix = *(unsigned int*)&value;
    radix ^= -(radix >> 31) | 0x80000000;
    return radix;
}

inline float RadixToFloat(unsigned int radix)
{
    radix ^= ((radix >> 31)-1) | 0x80000000;
    return *(float*)&radix;
}

void RadixSort8(float* fdata, int n)
{
	unsigned int* data = (unsigned int*)malloc(n * sizeof(unsigned int));
	unsigned int* sort = (unsigned int*)malloc(n * sizeof(unsigned int));

	// 4 histograms on the stack:
	const unsigned int kHist = 256;
	unsigned int b0[kHist * 4];
	unsigned int *b1 = b0 + kHist;
	unsigned int *b2 = b1 + kHist;
    unsigned int *b3 = b2 + kHist;

	for(int i = 0; i < kHist * 4; i++){
		b0[i] = 0;
	}
	for(int i = 0; i < n; i++){
		data[i] = FloatToRadix(fdata[i]);
		b0[data[i] & 0xFF]++;
		b1[(data[i] >> 8) & 0xFF]++;
		b2[(data[i] >> 16) & 0xFF]++;
        b3[data[i] >> 24]++;
	}
	
	// Sum the histograms -- each histogram entry records the number of values preceding itself.
    unsigned int sum0 = 0, sum1 = 0, sum2 = 0, sum3 = 0;
    unsigned int temp;
    for(int i = 0; i < kHist; i++){
        temp = b0[i] + sum0;
        b0[i] = sum0 - 1;
        sum0 = temp;

        temp = b1[i] + sum1;
        b1[i] = sum1 - 1;
        sum1 = temp;

        temp = b2[i] + sum2;
        b2[i] = sum2 - 1;
        sum2 = temp;

        temp = b3[i] + sum3;
        b3[i] = sum3 - 1;
        sum3 = temp;
    }

	// byte 0
	for(int i = 0; i < n; i++){
		// unsigned int radix = data[i];
		// unsigned int pos = radix & 0xFF;
		// sort[++b0[pos]] = radix;
        sort[++b0[data[i] & 0xFF]] = data[i];
	}

	// byte 1
	for(int i = 0; i < n; i++){
		// unsigned int radix = sort[i];
		// unsigned int pos = (radix >> 8) & 0xFF;
		// data[++b1[pos]] = radix;
        data[++b1[(sort[i] >> 8) & 0xFF]] = sort[i];
	}

	// byte 2
	for(int i = 0; i < n; i++){
		// unsigned int radix = data[i];
		// unsigned int pos = (radix >> 16) & 0xFF;
		// sort[++b2[pos]] = radix;
        sort[++b2[(data[i] >> 16) & 0xFF]] = data[i];
	}

    // byte 3
	for(int i = 0; i < n; i++){
		// unsigned int radix = sort[i];
		// unsigned int pos = radix >> 24;
        // fdata[++b3[pos]] = RadixToFloat(radix);
        fdata[++b3[sort[i] >> 24]] = RadixToFloat(sort[i]);
	}

	free(data);
	free(sort);
}

int main(int argc, char** argv) {
	if(argc != 4){
		fprintf(stderr, "must provide exactly 3 arguments!\n");
		return 1;
	}
    int n = atoi(argv[1]);
    int numtasks, rank, rc;
    rc = MPI_Init(&argc, &argv);
    if (rc != MPI_SUCCESS) {
        printf("Error starting MPI program. Terminating.\n");
        MPI_Abort(MPI_COMM_WORLD, rc);
    }
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    MPI_Comm comm_world;
    // if numtasks > n: use only n process
    if(numtasks > n){
        if(rank >= n){
            MPI_Comm_split(MPI_COMM_WORLD, MPI_UNDEFINED, rank, &comm_world);
        }
        else{
            MPI_Comm_split(MPI_COMM_WORLD, 0, rank, &comm_world);
        }
        // if unnecessary process: exit
        if(comm_world == MPI_COMM_NULL){
            // printf("Unnecessary process. Exit.\n"); // debug
            MPI_Finalize();
            return 0;
        }
        // get new numtasks & rank
        MPI_Comm_size(comm_world, &numtasks);
        MPI_Comm_rank(comm_world, &rank);
        // printf("this is rank %d in total %d tasks\n", rank, numtasks);  // debug  
    }
    else{
        comm_world = MPI_COMM_WORLD;
    }
    // printf("numtasks: %d\n", numtasks); // debug

    // ********************************************************
    // ** from now on we will use comm_world as communicator **
    // ********************************************************

    // calculate the number of element to read & offset
    int num = n / numtasks;
    int remainder = n % numtasks; 
    if(rank < remainder){
        num += 1;
    }
    int offset;
    if(rank < remainder){
        offset = rank * num;
    }
    else{
        offset = (num+1) * remainder + num * (rank - remainder);
    }
    // printf("this is rank %d, number of element to read %d, offset %d\n", rank, num, offset); // debug

    MPI_File input_file;
	MPI_File_open(comm_world, argv[2], MPI_MODE_RDONLY, MPI_INFO_NULL, &input_file);
    float *local_data = (float *)malloc(num * sizeof(float));
    MPI_File_read_at(input_file, offset * sizeof(float), local_data, num, MPI_FLOAT, MPI_STATUS_IGNORE);

    // sort the local data
    // std::sort(local_data, local_data+num);
    RadixSort8(local_data, num);
    // float_sort(local_data, local_data+num, rightshift());  //sorting in ascending order

    // *************************
    // ** start Odd-Even Sort **
    // *************************

    int num_next = (rank == remainder-1)? num-1: num;   // number of element of rank+1
    int num_prev = (rank == remainder)? num+1: num; // number of element of rank-1
    float *recv_data = (float *)malloc(std::max(num_next, num_prev) * sizeof(float));
    float *local_data_Even = (float *)malloc(num * sizeof(float));
    bool EVEN = (rank % 2 == 0);
    bool swap = false;
    bool total_swap = false;
    while(true){
        swap = false;
        // even phase
        if(EVEN){
            if(rank+1 != numtasks){ // skip last Even
                MPI_Sendrecv(local_data, num, MPI_FLOAT, rank+1, 0, recv_data, num_next, MPI_FLOAT, rank+1, MPI_ANY_TAG, comm_world, MPI_STATUS_IGNORE);
                if(local_data[num-1] > recv_data[0]){
                    for(int j = 0, k = 0; j+k < num; ){
                        if(k == num_next){
                            local_data_Even[j+k] = local_data[j];
                            j++;
                        }
                        else if(j == num){
                            local_data_Even[j+k] = recv_data[k];
                            k++;
                        }
                        else if(local_data[j] <= recv_data[k]){
                            local_data_Even[j+k] = local_data[j];
                            j++;
                        }
                        else{
                            local_data_Even[j+k] = recv_data[k];
                            k++;
                        }
                    }
                }
                else{
                    for(int j = 0; j < num; j++){
                        local_data_Even[j] = local_data[j];
                    }
                }
            }
            else{
                for(int j = 0; j < num; j++){
                    local_data_Even[j] = local_data[j];
                }
            }
        }
        else{
            MPI_Sendrecv(local_data, num, MPI_FLOAT, rank-1, 0, recv_data, num_prev, MPI_FLOAT, rank-1, MPI_ANY_TAG, comm_world, MPI_STATUS_IGNORE);
            if(local_data[0] < recv_data[num_prev-1]){
                for(int cnt = num-1, j = num_prev-1, k = num-1; cnt >= 0; cnt--){
                    if(local_data[k] >= recv_data[j]){
                        local_data_Even[cnt] = local_data[k];
                        k--;
                    }
                    else{
                        local_data_Even[cnt] = recv_data[j];
                        j--;
                    }
                }
            }
            else{
                for(int j = 0; j < num; j++){
                    local_data_Even[j] = local_data[j];
                }
            }
        }
        // Odd phase
        if(EVEN){
            if(rank != 0){  // skip 0
                MPI_Sendrecv(local_data_Even, num, MPI_FLOAT, rank-1, 0, recv_data, num_prev, MPI_FLOAT, rank-1, MPI_ANY_TAG, comm_world, MPI_STATUS_IGNORE);
                if(local_data_Even[0] < recv_data[num_prev-1]){
                    for(int cnt = num-1, j = num_prev-1, k = num-1; cnt >= 0; cnt--){
                        if(local_data_Even[k] >= recv_data[j]){
                            local_data[cnt] = local_data_Even[k];
                            k--;
                        }
                        else{
                            local_data[cnt] = recv_data[j];
                            j--;
                        }
                    }
                    swap = true;
                }
                else{
                    for(int j = 0; j < num; j++){
                        local_data[j] = local_data_Even[j];
                    }
                }
            }
            else{
                for(int j = 0; j < num; j++){
                    local_data[j] = local_data_Even[j];
                }
            }
        }
        else{
            if(rank+1 != numtasks){ // skip last Odd
                MPI_Sendrecv(local_data_Even, num, MPI_FLOAT, rank+1, 0, recv_data, num_next, MPI_FLOAT, rank+1, MPI_ANY_TAG, comm_world, MPI_STATUS_IGNORE);
                if(local_data_Even[num-1] > recv_data[0]){
                    for(int j = 0, k = 0; j+k < num; ){
                        if(k == num_next){
                            local_data[j+k] = local_data_Even[j];
                            j++;
                        }
                        else if(j == num){
                            local_data[j+k] = recv_data[k];
                            k++;
                        }
                        else if(local_data_Even[j] <= recv_data[k]){
                            local_data[j+k] = local_data_Even[j];
                            j++;
                        }
                        else{
                            local_data[j+k] = recv_data[k];
                            k++;
                        }
                    }
                    swap = true;
                }
                else{
                    for(int j = 0; j < num; j++){
                        local_data[j] = local_data_Even[j];
                    }
                }
            }
            else{
                for(int j = 0; j < num; j++){
                    local_data[j] = local_data_Even[j];
                }
            }
        }
        MPI_Allreduce(&swap, &total_swap, 1, MPI_CXX_BOOL, MPI_LOR, comm_world);
        if(total_swap == false){
            break;
        }
    }

    MPI_File output_file;
    int wc = MPI_File_open(comm_world, argv[3], MPI_MODE_CREATE|MPI_MODE_WRONLY, MPI_INFO_NULL, &output_file);
	MPI_File_write_at(output_file, offset * sizeof(float), local_data, num, MPI_FLOAT, MPI_STATUS_IGNORE);
    
	MPI_Finalize();
    return 0;
}