#include <omp.h>
#include <iostream>
#include <cstdlib>
#include <vector>
#include <math.h>

int main(int argc, char const *argv[])
{
    std::vector<int> numbers = {243,3,4,51,234,455,76,2,4326,78,643};
    int min_val = numbers[0];

    int thread_count = strtol(argv[1], NULL, 10);
    # pragma omp parallel num_threads(thread_count) reduction(min: min_val) 
    for (size_t i = 0; i < numbers.size(); ++i) {
        if (numbers[i] < min_val) {
            min_val = numbers[i];
        }
    }

    printf("Minimum = %u\n", min_val);
    return 0;
}
