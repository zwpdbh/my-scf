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

int ROWS = 12;
int COLS = 11;
int NUM_THREADS = 4;
int NUM_NODES = 4;

long s_in_each_node = 0;
float** mx = nullptr;
float* v = nullptr;
float* w = nullptr;
long s = 0;


typedef struct pthreadInfo {
    int thread_id;
    long from;
    long job_size;
    int row_in_mx;
    int p;
    pthread_t thread;
} pInfo;

void *emalloc(size_t s);
void *remalloc(void *p, size_t s);
void* thr_alloc_fn(void* arg);
void* thr_access_fn(void* arg);
void* thr_multiply_fn(void* arg);
void check_mx();
void check_w();
void construct_barrier(pInfo* pthreads);

pthread_t main_thread;

int main(int argc, char* argv[]) {

    s = ROWS * COLS;
    s_in_each_node = s / NUM_NODES;

    mx = (float**)emalloc(sizeof(float*) * NUM_NODES);

    pInfo* pthreads = (pInfo*) emalloc(sizeof(*pthreads) * NUM_NODES);
    pthread_t* thread; // array of pthread_t, for catching each created thread
    int err = 0;

    main_thread = pthread_self();
    printf("main thread is %lu\n", main_thread);

    /**Allocate m by n matrix with the shape NUM_NODES * some right size*/
    for (int i = 0; i < NUM_NODES; i++) {
        pthreads[i].thread_id = i;
        pthreads[i].row_in_mx = i;
        pthreads[i].job_size = (s/NUM_NODES);
        if (pthreads[i].thread_id == NUM_NODES - 1) {
            pthreads[i].job_size += (s % NUM_NODES);
        }
        err = pthread_create(&pthreads[i].thread, NULL, thr_alloc_fn, (void *)&pthreads[i]);
        if (err != 0) {
            printf("error during thread creation, exit..\n");
            exit(-1);
        }
    }

    /**manually created barrier*/
    construct_barrier(pthreads);
    check_mx();

    pthreads =(pInfo*)remalloc(pthreads, sizeof(*pthreads) * NUM_THREADS);
    for (int i = 0; i < NUM_THREADS; i++) {
        pthreads[i].thread_id = i;
        pthreads[i].row_in_mx = i / (NUM_THREADS / NUM_NODES);
        pthreads[i].p = i  - pthreads[i].row_in_mx * (NUM_THREADS / NUM_NODES);
        pthreads[i].job_size = s / NUM_THREADS;
        pthreads[i].from = pthreads[i].p * pthreads[i].job_size;
        if ((pthreads[i].p + 1) % (NUM_THREADS / NUM_NODES) == 0) {
            pthreads[i].job_size += ((s / NUM_NODES) % (NUM_THREADS / NUM_NODES));
        }
        if (pthreads[i].thread_id == NUM_THREADS - 1) {
            pthreads[i].job_size +=  (s % NUM_NODES);
        }

        err = pthread_create(&pthreads[i].thread, NULL, thr_access_fn, (void *)&pthreads[i]);
        if (err != 0) {
            printf("error during thread creation, exit...\n");
            exit(-1);
        }
    }

    construct_barrier(pthreads);
    printf("\n");
    check_mx();

//    /**do the multiplication with a vector*/
//    v = (float*)emalloc(sizeof(float) * COLS);
//    w = (float*)emalloc(sizeof(float) * ROWS);
//    for (int i = 0; i < COLS; i++) {
//        v[i] = 1;
//    }
//    for (int i = 0; i < ROWS; i++) {
//        w[i] = 0;
//    }
//
//    // distribute multiplication task on different threads
//    for (int i = 0; i < NUM_THREADS; i++) {
//        if (ROWS / NUM_THREADS == 0) {
//            pthreads[i].from = i * 1;
//            pthreads[i].job_size = 1;
//        } else {
//            pthreads[i].from  = i * (ROWS / NUM_THREADS);
//            pthreads[i].job_size = ROWS / NUM_THREADS;
//        }
//
//        if (i == NUM_THREADS - 1) {
//            pthreads[i].job_size += (ROWS % NUM_THREADS);
//        }
//        err = pthread_create(&pthreads[i].thread, NULL, thr_multiply_fn, (void*)&pthreads[i]);
//        if (err != 0) {
//            printf("error during thread creation, exit...\n");
//            exit(-1);
//        }
//    }
//
//    /**build barrier*/
//    construct_barrier(pthreads);
//    check_w();

    for (int i = 0; i < NUM_NODES; i++) {
        free(mx[i]);
    }
    free(mx);
    free(pthreads);
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

void *remalloc(void *p, size_t s) {
    void *result = realloc(p, s);
    if (result == NULL) {
        fprintf(stderr, "memory allocation failed");
        exit(EXIT_FAILURE);
    }
    return result;
}

// thread function to use libnuma to allocate memory on specific node
void* thr_alloc_fn(void* arg) {
    pInfo info = *(pInfo*)arg;
    *(mx + info.row_in_mx) = (float *)emalloc(info.job_size * sizeof(float));
    for (int k = 0; k < info.job_size; k++) {
        *(*(mx +info.row_in_mx) +k) = 0;
    }
    return (void *)0;
}


void* thr_access_fn(void* arg) {
    pInfo info = *(pInfo *)arg;
    for (long i = info.from; i < info.from + info.job_size; i++) {
        *(*(mx + info.row_in_mx) + i) = 3.0;
    }
    return (void *)0;
}

void* thr_multiply_fn(void* arg) {
//    eachInfo info = *(eachInfo*)arg;
    pInfo info = *(pInfo*)arg;
    for (int r = info.from; r < info.from  + info.job_size; ++r) {
        // get corresponding mx(i, j)
        for (int i = 0; i < COLS; i++) {
            long l = r * COLS + i;
            long row_in_mx = l / s_in_each_node - 1;
            if (row_in_mx < 0) {
                row_in_mx = 0;
            }
            long col_in_mx = l % (s / NUM_NODES);
            w[r] += (*(*(mx + row_in_mx) + col_in_mx) * v[i]);
        }
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

void check_w() {
    printf("w = \n");
    for (long i = 0; i < ROWS; i++) {
        printf("%.f ", w[i]);
    }
    printf("\n");
}

void construct_barrier(pInfo* pthreads) {
    int err;
    void* status;
    for (int i = 0; i < NUM_THREADS; i++) {
        if (pthreads[i].thread == main_thread) {
            printf("one of the child thread is main thread !!!\n");
        }
        err = pthread_join(pthreads[i].thread, &status);
        if (err) {
            printf("error, return code from pthread_join() is %d\n", *(int*)status);
        }
    }
}