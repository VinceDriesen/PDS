#include <stdio.h>
#include <pthread.h>
#include <math.h>
#include "timer.h"
#include <fstream>
#include <iostream>
#include <stdlib.h>

#define NUM_THREADS 4
#define AANTAL_REPEATS 100

double totaalResultaat = 0;
pthread_mutex_t mutex;

struct Resultaat {
    double timerResultaat;
    double eindResultaat;
};

double sqrtFunction(const double x) {
    return sqrt(1 - (x * x));
}

double berekenOppervlakte(const double x1, const double x2, double (*function)(double)) {
    double oppervlakte = (x2 - x1) * function(x1);
    return oppervlakte;
}

double totaalOppervlaktePerThread(const double x1, const double x2, const int intervalPerThread) {
    double difference = (x2 - x1) / intervalPerThread; 
    double start = x1;
    double end = x1 + difference;
    double resultaat = 0;
    for (int i = 0; i < intervalPerThread; i++) {
        // printf("Thread: van %f, %f, %i \n", start, end, intervalPerThread);
        resultaat += berekenOppervlakte(start, end, sqrtFunction);
        start = end;
        end += difference;
    }
    return resultaat;
}

typedef struct {
    double x1;
    double x2;
    int intervalPerThread;
    double resultaat;
} ThreadArgs13;

void* threadFunc13(void* arg) {
    ThreadArgs13* args = (ThreadArgs13*)arg; args->resultaat = totaalOppervlaktePerThread(args->x1, args->x2, args->intervalPerThread);
    return NULL;
}

Resultaat main13(int aantalIntervallen) {
    pthread_t subThreads[NUM_THREADS];
    ThreadArgs13 threadArgs[NUM_THREADS];

    double x1 = -1.0;
    double x2 = 1.0;

    double somTimerResultaat = 0;
    double somEindResultaat = 0;

    int intervallenPerThread = aantalIntervallen / NUM_THREADS;
    int rest = aantalIntervallen % NUM_THREADS;

    AutoAverageTimer t("timer");
    for(int repeat = 0; repeat < AANTAL_REPEATS; repeat++) {
        t.start();
        
        double huidigeX1 = x1;

        for (int i = 0; i < NUM_THREADS; i++) {
            int workload = intervallenPerThread;
            if (i < rest) {
                workload++;
            }
            double stap = (x2 - x1) * ((double)workload / aantalIntervallen);

            threadArgs[i].x1 = huidigeX1;
            threadArgs[i].x2 = huidigeX1 + stap;
            threadArgs[i].intervalPerThread = workload;
            
            huidigeX1 += stap;

            pthread_create(&subThreads[i], NULL, threadFunc13, &threadArgs[i]);
        }

        double eindResultaat = 0;
        for (int i = 0; i < NUM_THREADS; i++) {
            pthread_join(subThreads[i], NULL);
            eindResultaat += threadArgs[i].resultaat;
        }
        t.stop();
        somTimerResultaat += t.durationNanoSeconds();
        somEindResultaat += eindResultaat;
    }
    double gemiddeldeTimer = somTimerResultaat / AANTAL_REPEATS;
    double gemiddeldeResultaat = somEindResultaat / AANTAL_REPEATS;
    return {gemiddeldeTimer, gemiddeldeResultaat};
}


typedef struct {
    double x1;
    double x2;
    int intervalPerThread;
} ThreadArgs14;

void* threadFunc(void* arg) {
    ThreadArgs14* args = (ThreadArgs14*)arg;
    double resultaat = totaalOppervlaktePerThread(args->x1, args->x2, args->intervalPerThread);
    pthread_mutex_lock(&mutex);
    totaalResultaat += resultaat;
    pthread_mutex_unlock(&mutex);
    return NULL;
}

Resultaat main14(int aantalIntervallen) {
    pthread_t subThreads[NUM_THREADS];
    ThreadArgs14 threadArgs[NUM_THREADS];

    double x1 = -1.0;
    double x2 = 1.0;

    double somTimerResultaat = 0;
    double somEindResultaat = 0;

    int intervallenPerThread = aantalIntervallen / NUM_THREADS;
    int rest = aantalIntervallen % NUM_THREADS;

    AutoAverageTimer t("timer");
    pthread_mutex_init(&mutex, NULL);

    for (int repeat = 0; repeat < AANTAL_REPEATS; ++repeat) {
        totaalResultaat = 0;
        t.start();

        double huidigeX1 = x1;

        for (int i = 0; i < NUM_THREADS; i++) {
            int workload = intervallenPerThread;
            if (i < rest) {
                workload++;
            }
            double stap = (x2 - x1) * ((double)workload / aantalIntervallen);

            threadArgs[i].x1 = huidigeX1;
            threadArgs[i].x2 = huidigeX1 + stap;
            threadArgs[i].intervalPerThread = workload;
            
            huidigeX1 += stap;

            pthread_create(&subThreads[i], NULL, threadFunc, &threadArgs[i]);
        }

        for (int i = 0; i < NUM_THREADS; i++) {
            pthread_join(subThreads[i], NULL);
        }

        t.stop();
        somTimerResultaat += t.durationNanoSeconds();
        somEindResultaat += totaalResultaat;
    }
    
    pthread_mutex_destroy(&mutex);
    
    double gemiddeldeTimer = somTimerResultaat / AANTAL_REPEATS;
    double gemiddeldeResultaat = somEindResultaat / AANTAL_REPEATS;
    return {gemiddeldeTimer, gemiddeldeResultaat};
}


void metPlot() {
    std::ofstream out("resultaten.csv");
    out << "intervallen,tijd_main13_ns,tijd_main14_ns\n";

    int aantalKeerInterval = 28;

    for (int i = 0; i < aantalKeerInterval; ++i) {
        int aantal = 2 << i;
        Resultaat res13 = main13(aantal);
        Resultaat res14 = main14(aantal);

        out << aantal << "," << res13.timerResultaat << "," << res14.timerResultaat << "\n";
        std::cout << "Intervallen " << aantal 
                  << " -> tijd main13: " << res13.timerResultaat 
                  << " ns, tijd main14: " << res14.timerResultaat << " ns\n";
    }

    out.close();
    std::cout << "Resultaten geschreven naar resultaten.csv\n";
    std::cout << "Plotten...\n";
    system("uv run plot.py resultaten.csv plot.png");
}

int main(int argc, char *argv[]) {
    const bool plot = true;

    if (plot) {
        metPlot();
    } else {
        int aantalIntervallen = 20;
        Resultaat res13 = main13(aantalIntervallen);
        Resultaat res14 = main14(aantalIntervallen);

        printf("Oef 13: Totaal oppervlakte: %f met tijd: %f ns\n", res13.eindResultaat, res13.timerResultaat);
        printf("Oef 14: Totaal oppervlakte: %f met tijd: %f ns\n", res14.eindResultaat, res14.timerResultaat);
    }

    return 0;
}
