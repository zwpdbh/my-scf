//
// Created by zwpdbh on 07/05/2018.
//

#include <pthread.h>
#include <iostream>
#include <chrono>
#include <sched.h>
#include <sys/types.h>
#include <unistd.h>
#include <sched.h>

using namespace std;

void *emalloc(size_t s);
void* thr_alloc_fn(void* arg);
void* thr_access_fn(void* arg);
void check_mx();

#define ROWS 7
#define COLS 11
#define NUM_THREADS 8
#define NUM_NODES 8

long s_in_each_node = 0;
float** mx = nullptr;
long s = 0;

typedef struct Access_info_per_thread pInfo;
struct Access_info_per_thread {
    // hold thread_id passed from index in the loop
    int thread_id;
    // hold the size, for each thread
    long job_size;
};

int main(int argc, char* argv[]) {
    s = ROWS * COLS;
    s_in_each_node = s / NUM_NODES;

    mx = (float**)emalloc(sizeof(float*) * NUM_NODES);

    int* tid;   // array of int, for passing index into each thread
    pthread_t* thread; // array of pthread_t, for catching each created thread
    int err = 0;
    void* status;
    tid = (int*)emalloc(sizeof(*tid) * NUM_NODES);
    thread = (pthread_t*)emalloc(sizeof(*thread) * NUM_NODES);

    /**Allocate m by n matrix with the shape NUM_NODES * some right size*/
    for (int i = 0; i < NUM_NODES; i++) {
        tid[i] = i;
        err = pthread_create(&thread[i], nullptr, thr_alloc_fn, (void *)&tid[i]);
        if (err != 0) {
            printf("error during thread creation, exit..\n");
            exit(-1);
        }
    }

    /**manually created barrier*/
    for (int i = 0; i < NUM_NODES; i++) {
        err = pthread_join(thread[i], &status);
        if (err) {
            printf("error, return code from pthread_join() is %d\n", *(int *)status);
        }
    }

    check_mx();
    free(tid);

    pInfo* threadInfo;
    // for catching each created threads
    free(thread);
    thread = (pthread_t*) emalloc(sizeof(*thread) * NUM_THREADS);
    // for passing useful partition info into each thread
    threadInfo = (pInfo*) emalloc(sizeof(*threadInfo) * NUM_THREADS);

    for (int i = 0; i < NUM_THREADS; i++) {
        threadInfo[i].thread_id = i;
        threadInfo[i].job_size = s / NUM_THREADS;
        err = pthread_create(&thread[i], nullptr, thr_access_fn, (void *)&threadInfo[i]);
        if (err != 0) {
            printf("error during thread creation, exit...\n");
            exit(-1);
        }
    }

    // make sure all thread finished the execution
    for (int i = 0; i < NUM_THREADS; i++) {
        err = pthread_join(thread[i], &status);
        if (err) {
            printf("error, return code from pthread_join() is %d\n", *(int*)status);
        }
    }

    printf("\n");
    check_mx();



    for (int i = 0; i < NUM_NODES; i++) {
        free(mx[i]);
    }
    free(mx);
    free(threadInfo);
    free(thread);

    exit(0);
}

void *emalloc(size_t s) {
    void *result = malloc(s);
    if (result == nullptr) {
        fprintf(stderr, "memory allocation failed");
        exit(EXIT_FAILURE);
    }
    return result;
}

// thread function to use libnuma to allocate memory on specific node
void* thr_alloc_fn(void* arg) {
    pid_t pid;
    pthread_t tid;
    int i;

    pid = getpid();
    tid = pthread_self();
    i = *(int *)arg;

    printf("pid = %lu, tid = %lu, for i = %d\n", (unsigned long)pid, (unsigned long)tid, i);
    mx[i] = (float*)emalloc(s_in_each_node * sizeof(float));

    long size_for_each_node = s_in_each_node;
    if (i == NUM_NODES -1) {
        size_for_each_node += s % NUM_NODES;
        printf("for thread = %d, it gets extra job, size_for_each_node = %ld\n", i, size_for_each_node);
    }
    for (int k = 0; k < size_for_each_node; k++) {
        *(*(mx + i) + k) = 0;
    }

    return 0;
}


void* thr_access_fn(void* arg) {
    pInfo info;
    info = *(struct Access_info_per_thread*)arg;

    /**node_index tells which node the current thread is executing in
     * It depends on how many threads one node could hold on*/
    int node_index = info.thread_id / (NUM_THREADS / NUM_NODES);

    /**Only know which node this thread is executing on is not enough,
     * since there are multiple thread working on the same node section.
     * So we need to know which part of the section in the node is the current thread operate on*/
    int p = info.thread_id - node_index * (NUM_THREADS / NUM_NODES);

    long from = p * info.job_size;
    long to = from + info.job_size;

    // if it is the last thread in a NUMA node section
    if ((p + 1) % (NUM_THREADS / NUM_NODES) == 0) {
        to += s_in_each_node % (NUM_THREADS / NUM_NODES);
    }

    // if it is the last thread
    if (info.thread_id == NUM_THREADS - 1) {
        to = to + s % NUM_NODES;
    }

    for (long i = from; i < to; i++) {
        *(*(mx + node_index) + i) = 3.0;
    }

    return (void *)0;
}

void check_mx() {
    // check the content of mx
    long size_for_each_node = s_in_each_node;

    for(int i = 0; i < NUM_NODES; i++) {
        if (i == NUM_NODES -1) {
            size_for_each_node += s % NUM_NODES;
        }
        for (int j = 0; j < size_for_each_node; j++) {
            printf("%.f ", *(*(mx + i) + j));
        }
        printf("\n");
    }
}