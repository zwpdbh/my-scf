//
// Created by zwpdbh on 13/05/2018.
//

#include <omp.h>
#include <stdio.h>
int main(int argc, char* argv[]) {

#pragma omp parallel proc_bind(close) num_threads(8) default(none)
  {
    printf("ok\n");
  }

    return 0;
}