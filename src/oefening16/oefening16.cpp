#include <omp.h>
#include <iostream>
#include <cstdlib>

void printHelloWorld() {
    int my_rank = omp_get_thread_num();
    int thread_count = omp_get_num_threads();
    printf("Hello, World! van thread %d van %d\n", my_rank, thread_count);
}

int main(int argc, char *argv[]) {
    int thread_count = strtol(argv[1], NULL, 10);

    #pragma omp parallel num_threads(thread_count)
    {
        printHelloWorld();
    }
    return 0;
}
