#include <iostream>
#include <fstream>
#include <string>
#include <map>
#include <vector>
#include <algorithm>
#include <cmath>
#include "mpi.h"
#include <cstring>
#include <deque>
#include <unistd.h>
#include <queue>
#include "Map.h"
#include "Reduce.h"

using namespace std;

// parallel
int nodes;
int rank_id;
int num_threads;
// arguments
string job_name;
int num_reducer;
int delay;
string input_filename;
int chunk_size;
string locality_config_filename;
string output_dir;
// global
int num_chunk;
// thread pool
queue<pair<int, int>> tasks;    // (chunkID, location)
pthread_mutex_t mutex;
pthread_cond_t cond;
// mapper task
queue<pair<int, pair<int, int>>> complete;  // (chunkID, (time, pairs))
pthread_mutex_t mutex_complete;
pthread_cond_t cond_complete;
int num_jobs;   // number of jobs that are dispatched but not finished

int calc_time(struct timespec start_time, struct timespec end_time)
{
    struct timespec temp;
    if ((end_time.tv_nsec - start_time.tv_nsec) < 0) {
        temp.tv_sec = end_time.tv_sec-start_time.tv_sec-1;
        temp.tv_nsec = 1000000000 + end_time.tv_nsec - start_time.tv_nsec;
    } else {
        temp.tv_sec = end_time.tv_sec - start_time.tv_sec;
        temp.tv_nsec = end_time.tv_nsec - start_time.tv_nsec;
    }
    double exe_time = temp.tv_sec + (double) temp.tv_nsec / 1000000000.0;
    return (int)(exe_time+0.5);
}

void jobtracker()
{
    struct timespec start_time, end_time;
    clock_gettime(CLOCK_MONOTONIC, &start_time);

    ofstream out(output_dir + job_name + "-log.out");
    out << time(nullptr) << ",Start_Job," << nodes << "," << num_threads << "," << job_name << "," << num_reducer << "," << delay << "," << input_filename << "," << chunk_size << "," << locality_config_filename << "," << output_dir << endl;

    deque<pair<int, int>> jobs;   // job queue
    int request_node;
    int job_send[2];    // (chunkID, nodeID)
    int cnt;
    int complete_info[3];   // (job, time, pairs)
    int total_pairs = 0;

    /////////
    // Map //
    /////////

    ifstream input_file(locality_config_filename);
    string line;
    while (getline(input_file, line)) {
        size_t pos = line.find(" ");
        int chunkID = stoi(line.substr(0, pos));
        int nodeID = stoi(line.substr(pos+1)) % (nodes-1);
        jobs.push_back(make_pair(chunkID, nodeID));    // Locality information: (chunkID, nodeID)
    }
    num_chunk = jobs.size();

    while (!jobs.empty()) {
        MPI_Recv(&request_node, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for (auto job = jobs.begin(); job != jobs.end(); job++) {
            if (job->second == request_node) {  // Locality-Aware Scheduling Algorithm
                job_send[0] = job->first;
                job_send[1] = job->second;
                out << time(nullptr) << ",Dispatch_MapTask," << job_send[0] << "," << request_node << endl;
                MPI_Send(&job_send, 2, MPI_INT, request_node, 1, MPI_COMM_WORLD);
                jobs.erase(job);
                break;
            }
            if (job == prev(jobs.end())) {  // if no match: FIFO
                job_send[0] = jobs.begin()->first;
                job_send[1] = jobs.begin()->second;
                out << time(nullptr) << ",Dispatch_MapTask," << job_send[0] << "," << request_node << endl;
                MPI_Send(&job_send, 2, MPI_INT, request_node, 1, MPI_COMM_WORLD);
                jobs.erase(jobs.begin());
                break;
            }
        }
    }

    // notify workers that all jobs are dispatched
    cnt = 0;
    job_send[0] = -1;
    job_send[1] = -1;
    while (cnt != (nodes-1)) {{
        MPI_Recv(&request_node, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        cnt++;
        MPI_Send(&job_send, 2, MPI_INT, request_node, 1, MPI_COMM_WORLD);
    }}

    // receive complete information
    cnt = 0;
    while (cnt != num_chunk) {{
        MPI_Recv(&complete_info, 3, MPI_INT, MPI_ANY_SOURCE, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        cnt++;
        total_pairs += complete_info[2];
        out << time(nullptr) << ",Complete_MapTask," << complete_info[0] << "," << complete_info[1] << endl;
    }}

    /////////////
    // shuffle //
    /////////////

    out << time(nullptr) << ",Start_Shuffle," << to_string(total_pairs) << endl;

    struct timespec start_shuffle_time, end_shuffle_time;
    clock_gettime(CLOCK_MONOTONIC, &start_shuffle_time);

    ofstream *files = new ofstream[num_reducer];
    for (int i = 0; i < num_reducer; i++) {
        files[i] = ofstream(output_dir + job_name + "-intermediate_reducer_" + to_string(i+1) + ".out");
    }
    vector<Item> data;
    for (int i = 0; i < num_chunk; i++) {
        data.clear();
        ifstream input_file(output_dir + job_name + "-intermediate" + to_string(i+1) + ".out");
        string line;
        while (getline(input_file, line)) {
            size_t pos = line.find(" ");
            string key = line.substr(0, pos);
            int value = stoi(line.substr(pos+1));
            data.push_back(make_pair(key, value));
        }
        for (auto it: data) {
            int idx = Partition(it.first, num_reducer);
            files[idx] << it.first << " " << it.second << endl;
        }
    }
    delete [] files;
    
    // int x = rand() % 5;
    // cout << x << endl;
    // sleep(x);
    clock_gettime(CLOCK_MONOTONIC, &end_shuffle_time);
    out << time(nullptr) << ",Finish_Shuffle," << to_string(calc_time(start_shuffle_time, end_shuffle_time)) << endl;

    ////////////
    // Reduce //
    ////////////

    cnt = num_reducer;
    while (cnt > 0) {
        MPI_Recv(&request_node, 1, MPI_INT, MPI_ANY_SOURCE, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        job_send[0] = cnt;
        out << time(nullptr) << ",Dispatch_ReduceTask," << job_send[0] << "," << request_node << endl;
        MPI_Send(&job_send, 1, MPI_INT, request_node, 4, MPI_COMM_WORLD);
        cnt--;
    }

    // notify workers that all jobs are dispatched
    cnt = 0;
    job_send[0] = -1;
    while (cnt != (nodes-1)) {{
        MPI_Recv(&request_node, 1, MPI_INT, MPI_ANY_SOURCE, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        cnt++;
        MPI_Send(&job_send, 1, MPI_INT, request_node, 4, MPI_COMM_WORLD);
    }}

    // receive complete information
    cnt = 0;
    while (cnt != num_reducer) {{
        MPI_Recv(&complete_info, 2, MPI_INT, MPI_ANY_SOURCE, 5, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        cnt++;
        out << time(nullptr) << ",Complete_ReduceTask," << complete_info[0] << "," << complete_info[1] << endl;
    }}

    clock_gettime(CLOCK_MONOTONIC, &end_time);
    out << time(nullptr) << ",Finish_Job," << to_string(calc_time(start_time, end_time)) << endl;
}

void submitTask(pair<int, int> job) 
{
    pthread_mutex_lock(&mutex);
    tasks.push(job);
    pthread_mutex_unlock(&mutex);
    pthread_cond_signal(&cond);
}

void* pool(void* args) 
{
    struct timespec start_time, end_time, temp;
    double exe_time;

    while (1) {
        pair<int, int> task;

        // get task
        pthread_mutex_lock(&mutex);
        while (tasks.empty()) {
            pthread_cond_wait(&cond, &mutex);
        }
        task = tasks.front();
        tasks.pop();
        pthread_mutex_unlock(&mutex);

        clock_gettime(CLOCK_MONOTONIC, &start_time);

        // if read from a remote location
        if (task.second != rank_id) {
            sleep(delay);
        }

        // Input split function
        map<int, string> records = Input_split(task.first, input_filename, chunk_size);

        // Map function
        map<string, int> map_output;
        for (auto record: records) {
            map<string, int> map_result = Map(record);
            for (auto it: map_result) {
                if (map_output.count(it.first) == 0) {
                    map_output[it.first] = it.second;
                }
                else {
                    map_output[it.first] += it.second;
                }
            }
        }

        // Write intermediate result
        ofstream out(output_dir + job_name + "-intermediate" + to_string(task.first) + ".out");
        for (auto it: map_output) {
            out << it.first << " " << it.second << endl;
        }

        // int x = rand() % 5;
        // cout << task.first << " " << x << endl;
        // sleep(x);
        clock_gettime(CLOCK_MONOTONIC, &end_time);

        pthread_mutex_lock(&mutex_complete);
        complete.push(make_pair(task.first, make_pair(calc_time(start_time, end_time), map_output.size())));
        num_jobs--;
        pthread_mutex_unlock(&mutex_complete);
        pthread_cond_signal(&cond_complete);
    }
}

void tasktracker()
{
    /////////
    // Map //
    /////////

    pthread_t threads[num_threads-1];
    pthread_mutex_init(&mutex, NULL);
    pthread_cond_init(&cond, NULL);
    pthread_mutex_init(&mutex_complete, NULL);
    pthread_cond_init(&cond_complete, NULL);
    for (int i = 0; i < num_threads-1; i++) {
        pthread_create(&threads[i], NULL, &pool, NULL);
    }

    num_jobs = 0;

    int job[2] = {0};
    while (true) {
        pthread_mutex_lock(&mutex_complete);
        while (num_jobs == num_threads-1) {
            pthread_cond_wait(&cond_complete, &mutex_complete);
        }
        pthread_mutex_unlock(&mutex_complete);
        MPI_Send(&rank_id, 1, MPI_INT, (nodes-1), 0, MPI_COMM_WORLD);
        MPI_Recv(&job, 2, MPI_INT, (nodes-1), 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (job[0] == -1) { // termination condition, all tasks are dispatched
            break;
        }
        pthread_mutex_lock(&mutex_complete);
        num_jobs++;
        pthread_mutex_unlock(&mutex_complete);
        cout << "worker" << rank_id << " get job" << job[0] << " stored at device" << job[1] << endl;
        submitTask(make_pair(job[0], job[1]));
        usleep(200);
    }

    int complete_info[3];
    while (true) {
        pthread_mutex_lock(&mutex_complete);
        if (complete.empty() && num_jobs == 0) {    // if all jobs are finished and all information is sent
            pthread_mutex_unlock(&mutex_complete);
            break;
        }
        else if (!complete.empty()) {   // send information
            pair<int, pair<int, int>> info = complete.front();
            complete.pop();
            pthread_mutex_unlock(&mutex_complete);
            complete_info[0] = info.first;
            complete_info[1] = info.second.first;
            complete_info[2] = info.second.second;
            MPI_Send(&complete_info, 3, MPI_INT, (nodes-1), 2, MPI_COMM_WORLD);
        }
        else {
            pthread_mutex_unlock(&mutex_complete);
        }
    }

    ////////////
    // Reduce //
    ////////////

    struct timespec start_time, end_time, temp;
    double exe_time;
    queue<pair<int, int>> reduce_job_time;
    while (true) {
        MPI_Send(&rank_id, 1, MPI_INT, (nodes-1), 3, MPI_COMM_WORLD);
        MPI_Recv(&job, 1, MPI_INT, (nodes-1), 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (job[0] == -1) { // termination condition, all tasks are dispatched
            break;
        }
        cout << "worker" << rank_id << " get reduce job" << job[0] << endl;
        
        clock_gettime(CLOCK_MONOTONIC, &start_time);

        vector<Item> data;
        string intermediate_file = output_dir + job_name + "-intermediate_reducer_" + to_string(job[0]) + ".out";
        ifstream input_file(intermediate_file);
        string line;
        while (getline(input_file, line)) {
            size_t pos = line.find(" ");
            string key = line.substr(0, pos);
            int value = stoi(line.substr(pos+1));
            data.push_back(make_pair(key, value));
        }

        data = Sort(data);

        map<string, vector<int>, classcomp> grouped_data = Group(data);

        vector<Item> reduce_result;
        for (auto it: grouped_data) {
            reduce_result.push_back(Reduce(it.first, it.second));
        }

        Output(reduce_result, output_dir, job_name, job[0]);

        // int x = rand() % 5;
        // cout << job[0] << " " << x << endl;
        // sleep(x);
        clock_gettime(CLOCK_MONOTONIC, &end_time);
        reduce_job_time.push(make_pair(job[0], calc_time(start_time, end_time)));
    }

    while (!reduce_job_time.empty()) {
        pair<int, int> info = reduce_job_time.front();
        reduce_job_time.pop();
        complete_info[0] = info.first;
        complete_info[1] = info.second;
        MPI_Send(&complete_info, 2, MPI_INT, (nodes-1), 5, MPI_COMM_WORLD);
    }
}

int main(int argc, char **argv)
{
    // get arguments
    if(argc != 8){
		fprintf(stderr, "must provide exactly 7 arguments!\n");
		return 1;
	}
    job_name = string(argv[1]);
    num_reducer = stoi(argv[2]);
    delay = stoi(argv[3]);
    input_filename = string(argv[4]);
    chunk_size = stoi(argv[5]);
    locality_config_filename = string(argv[6]);
    output_dir = string(argv[7]);

    // MPI initialize
    int rc;
    rc = MPI_Init(&argc, &argv);
    if (rc != MPI_SUCCESS) {
        printf("Error starting MPI program. Terminating.\n");
        MPI_Abort(MPI_COMM_WORLD, rc);
    }
    MPI_Comm_size(MPI_COMM_WORLD, &nodes);
    // number of workers: nodes-1
    // master ID: nodes-1
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_id);

    // thread initialize
    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    num_threads = CPU_COUNT(&cpu_set);

    // start MapReduce
    if (rank_id == (nodes-1)) {
        jobtracker();
    }
    else {
        tasktracker();
    }

    MPI_Finalize();
    return 0;
}