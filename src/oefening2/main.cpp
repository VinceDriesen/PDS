#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include "timer.h"

long long som1(std::vector<long long> &array) {
    long long som = 0;
    int n = std::sqrt(array.size());
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            som += array[i * n + j];
        }
    }
    return som;
}

long long som2(std::vector<long long> &array) {
    long long som = 0;
    int n = std::sqrt(array.size());
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            som += array[i * n + j];
        }
    }
    return som;
}

void time(std::vector<long long> &array, int amountOfRuns) {
    {
        AutoAverageTimer t("som1");
        for (int i = 0; i < amountOfRuns; ++i) {
            t.start(); 
            std::cout << som1(array) << std::endl;
            t.stop();
        }
        t.report();
    }
    {
        AutoAverageTimer t("som2");
        for (int i = 0; i < amountOfRuns; ++i) {
            t.start(); 
            std::cout << som2(array) << std::endl;
            t.stop();
        }
        t.report();
    }
}

int main() {
    std::cout << "Oefening 2" << std::endl;

    const long long n = 20000;
    std::vector<long long> array(n * n);
    for (long long i = 0; i < n * n; ++i) {
        array[i] = i % 100;
    }
    int amountOfRuns = 5;
    time(array, amountOfRuns);


    return 0;
}
