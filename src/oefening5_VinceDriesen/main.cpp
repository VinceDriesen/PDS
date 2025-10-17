#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <cstdlib>
#include "timer.h"

struct Entry {
    Entry *next;
    uint64_t padding[127];
};

double loopArray(Entry* start, const uint64_t& aantalSteps) {
    Entry* current = start;

    AutoAverageTimer t("Entries");
    t.start();

    for (uint64_t i = 0; i < aantalSteps; ++i) {
        current = current->next;
    }

    t.stop();

    double avgNs = t.durationNanoSeconds() / static_cast<double>(aantalSteps);
    return avgNs;
}

void metPlot() {
    std::cout << "Oefening 5" << std::endl;

    int maxEntries;
    std::cout << "Geef het maximum aantal entries: ";
    std::cin >> maxEntries;

    const uint64_t aantalSteps = 20000000;
    const int numTests = 100;

    std::ofstream out("data.csv");
    out << "entries,size_MB,avg_ns\n";

    for (int i = 0; i < numTests; i++) {
        double fraction = static_cast<double>(i) / (numTests - 1);
        int n = static_cast<int>(std::pow(10.0, fraction * std::log10(maxEntries)));
        if (n < 1) n = 1;

        std::vector<Entry> entries(n);
        for (size_t j = 0; j < entries.size(); ++j) {
            entries[j].next = &entries[(j + 1) % entries.size()];
        }

        double avgNs = loopArray(&entries[0], aantalSteps);
        double sizeMB = (n * sizeof(Entry)) / (1024.0 * 1024.0);

        out << n << "," << sizeMB << "," << avgNs << "\n";
    }

    out.close();

    std::cout << "Resultaten geschreven naar data.csv\n";
    std::cout << "Plotten...\n";

    system("python3 src/oefening5/main.py data.csv src/oefening5/plot.png");
}

int main() {
    const bool plot = true;

    if (plot)
    {
        metPlot();
    }

    else {
        std::cout << "Oefening 5 zonder plot" << std::endl;

        std::cout << "Geef het aantal entries: ";
        int n;
        std::cin >> n;
        std::cout << "Aantal entries: " << n << std::endl;

        std::vector<Entry> entries(n);
        for (size_t j = 0; j < entries.size(); ++j) {
            entries[j].next = &entries[(j + 1) % entries.size()];
        }
        const uint64_t aantalSteps = 20000000;
        double avgNs = loopArray(&entries[0], aantalSteps);

        std::cout << "Gemiddelde tijd per stap: " << avgNs << " ns" << std::endl;
    }

    return 0;
}
