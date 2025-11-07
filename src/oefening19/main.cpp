#include <omp.h>
#include <iostream>
#include <cmath>
#include <string>

bool is_prime(size_t number)
{
    if (number < 2) return false;
    size_t to_check = std::sqrt(number) + 1;
    for (size_t i = 2; i < to_check; ++i)
    {
        if (number % i == 0)
            return false;
    }
    return true;
}

void forLoop(const std::string& sched, const int min, const int max) {
    int count = 0;
    double start = omp_get_wtime();

    if (sched == "static")
    {
        #pragma omp parallel for reduction(+:count) schedule(static)
        for (size_t i = min; i <= max; ++i)
            if (is_prime(i)) count++;
    }
    else if (sched == "dynamic")
    {
        #pragma omp parallel for reduction(+:count) schedule(dynamic)
        for (size_t i = min; i <= max; ++i)
            if (is_prime(i)) count++;
    }
    else if (sched == "guided")
    {
        #pragma omp parallel for reduction(+:count) schedule(guided)
        for (size_t i = min; i <= max; ++i)
            if (is_prime(i)) count++;
    }

    double end = omp_get_wtime();
    std::cout << "Schedule: " << sched 
              << " | Tijd: " << (end - start) 
              << "s | Aantal priemgetallen: " << count << "\n";
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cerr << "Gebruik: " << argv[0] << " <aantal_threads>\n";
        return 1;
    }

    int thread_count = strtol(argv[1], NULL, 10);
    omp_set_num_threads(thread_count);

    const int lowest = 2;
    const int highest = 100'000'000; // 10^8

    std::cout << "Aantal threads: " << thread_count << "\n";

    for (const std::string& sched : {"static", "dynamic", "guided"}) {
        forLoop(sched, lowest, highest);
    }

    return 0;
}
