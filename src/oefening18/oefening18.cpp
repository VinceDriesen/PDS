#include <omp.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cstdint>
#include <cmath>
#include <cstdlib>
#include "timer.h"

class BufferClass {
public:
    const int64_t g_numLoops = 1 << 27;
    void f(uint8_t *pBuffer, int offset) {
        for (int64_t i = 0; i < g_numLoops; i++) {
            pBuffer[offset] += 1;
        }
    }
};

void metPlot() {
    const int thread_count = 4;
    BufferClass bufferClass{};
    std::ofstream out("false_sharing.csv");
    out << "multiplier,size_bytes,avg_ns\n";

    std::cout << "False sharing test loopt...\n";

    for (int n = 0; n <= 20; n++) {   // tot multiplier 256
        int multiplier = 1 << n;     // 1, 2, 4, 8, ..., 256
        int buffer_size = multiplier * thread_count + 1;
        std::vector<uint8_t> buffer(buffer_size, 0);

        AutoAverageTimer t("false_sharing");
        t.start();

        #pragma omp parallel num_threads(thread_count)
        {
            int my_rank = omp_get_thread_num();
            int offset = multiplier * my_rank;
            bufferClass.f(buffer.data(), offset);
        }

        t.stop();
        double duration = t.durationNanoSeconds();

        out << multiplier << "," << buffer_size << "," << duration << "\n";

        std::cout << "Multiplier " << multiplier 
                  << " -> tijd " << duration << " ns\n";
    }

    out.close();

    std::cout << "Resultaten geschreven naar false_sharing.csv\n";
    std::cout << "Plotten...\n";

    system("uv run plot.py false_sharing.csv plot.png");
}

int main(int argc, char *argv[]) {
    const bool plot = true;

    if (plot) {
        metPlot();
    } else {
        std::cout << "Oefening zonder plot.\n";
    }

    return 0;
}
