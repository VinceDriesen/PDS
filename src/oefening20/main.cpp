#include <iostream>
#include <fstream>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <omp.h>

using namespace std;

vector<float> readFloats(const string &fname);

pair<float, float> findMinMax(const vector<float> &data) {
    float minVal = data[0];
    float maxVal = data[0];

    #pragma omp parallel for reduction(min:minVal) reduction(max:maxVal)
    for (size_t i = 0; i < data.size(); ++i) {
        if (data[i] < minVal) minVal = data[i];
        if (data[i] > maxVal) maxVal = data[i];
    }
    return {minVal, maxVal};
}

void histogram(int aantalBins, const vector<float> &histValues, float minValue, float maxValue) {
    vector<int> globalHist(aantalBins, 0);
    float binWidth = (maxValue - minValue) / aantalBins;

    #pragma omp parallel
    {
        vector<int> localHist(aantalBins, 0);

        #pragma omp for nowait
        for (size_t i = 0; i < histValues.size(); ++i) {
            float v = histValues[i];
            int bin = static_cast<int>((v - minValue) / binWidth);
            if (bin == aantalBins) bin--;
            localHist[bin]++;
        }

        #pragma omp critical
        {
            for (int i = 0; i < aantalBins; ++i)
                globalHist[i] += localHist[i];
        }
    }

    cout << "Histogram (" << aantalBins << " bins):" << endl;
    for (int i = 0; i < aantalBins; ++i) {
        float binStart = minValue + i * binWidth;
        float binEnd = binStart + binWidth;
        cout << "[" << binStart << ", " << binEnd << "]: " << globalHist[i] << endl;
    }
}

int main(int argc, char* argv[]) {
    int aantalBins;
    cout << "Oefening 20" << endl;
    cout << "Geef aantal bins: ";
    cin >> aantalBins;
    cout << "Aantal bins: " << aantalBins << endl;

    if (argc > 1) {
        int threads = strtol(argv[1], NULL, 10);
        omp_set_num_threads(threads);
        cout << "Aantal threads ingesteld op: " << threads << endl;
    }

    vector<float> histValues = readFloats("histvalues.dat");

    double start = omp_get_wtime();
    auto [minValue, maxValue] = findMinMax(histValues);
    histogram(aantalBins, histValues, minValue, maxValue);
    double end = omp_get_wtime();

    cout << "Totale uitvoeringstijd: " << (end - start) << " seconden" << endl;
    return 0;
}

vector<float> readFloats(const string &fname)
{
    vector<float> data;
    ifstream f(fname, ios::binary);

    f.seekg(0, ios_base::end);
    int pos = f.tellg();
    f.seekg(0, ios_base::beg);
    if (pos <= 0)
        throw runtime_error("File " + fname + " is leeg of onleesbaar");

    if (pos % sizeof(float) != 0)
        throw runtime_error("Bestand bevat geen volledig aantal floats");

    int num = pos / sizeof(float);
    data.resize(num);

    f.read(reinterpret_cast<char*>(data.data()), pos);
    if (f.gcount() != pos)
        throw runtime_error("Incompleet gelezen bestand");
    return data;
}
