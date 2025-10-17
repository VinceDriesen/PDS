#include <mpi.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <stdexcept>
#include <algorithm>

using namespace std;

vector<float> readFloats(const string &fname);

std::pair<float, float> findMinMax(const vector<float> &data) {
    float minVal = *std::min_element(data.begin(), data.end());
    float maxVal = *std::max_element(data.begin(), data.end());
    return {minVal, maxVal};
}


void printHistogram(int aantalBins, float minValue, float maxValue, float binWidth, std::vector<int> histogram) {
    std::cout << "Histogram (" << aantalBins << " bins):" << std::endl;

    for (int i = 0; i < aantalBins; ++i) {
        float binStart = minValue + i * binWidth;
        float binEnd = binStart + binWidth;

        std::cout << "[" << binStart << ", " << binEnd << "]: " << histogram[i] << std::endl;
    }
}

std::pair<float, float> calculateMinMax(int rank, const std::vector<float>& dataValues) {
    auto [localMin, localMax] = findMinMax(dataValues);
    float globalMin, globalMax;
    MPI_Allreduce(&localMin, &globalMin, 1, MPI_FLOAT, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&localMax, &globalMax, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
    return {globalMin, globalMax};
}

std::vector<float> getHistValues(int rank, int size) {
    std::vector<float> histValues;
    long chunkSize = 0;
    if (rank == 0) {
        histValues = readFloats("./histvalues.dat");
        long dataSize = histValues.size();
        if (dataSize % size != 0) {
            std::cerr << "Data size is not divisible by number of processes" << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        chunkSize = dataSize / size;
    }
    MPI_Bcast(&chunkSize, 1, MPI_LONG, 0, MPI_COMM_WORLD);
    std::vector<float> dataValues(chunkSize);
    MPI_Scatter(rank == 0 ? histValues.data() : nullptr, chunkSize, MPI_FLOAT,
                dataValues.data(), chunkSize, MPI_FLOAT, 0, MPI_COMM_WORLD);
    return dataValues;
}

void makeHistogram(int rank, int size, int aantalBins) {
    auto dataValues = getHistValues(rank, size);
    auto [globalMin, globalMax] = calculateMinMax(rank, dataValues);
    std::vector<int> localHistogram(aantalBins, 0);

    float binWidth = (globalMax - globalMin) / aantalBins;

    for (float v : dataValues) {
        int bin = static_cast<int>((v - globalMin) / binWidth);
        if (bin == aantalBins) bin--;
        localHistogram[bin]++;
    }
    std::vector<int> globalHistogram(aantalBins, 0);
    MPI_Reduce(localHistogram.data(), globalHistogram.data(), aantalBins, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) printHistogram(aantalBins, globalMin, globalMax, binWidth, globalHistogram);
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int aantalBins;
    if (rank == 0) {
        std::cout << "Oefening 11" << std::endl;
        std::cout << "Geef aantal bins: " << std::endl;
        std::cin >> aantalBins;
        std::cout << "Aantal bins zijn: " << aantalBins << std::endl;
    }

    MPI_Bcast(&aantalBins, 1, MPI_INT, 0, MPI_COMM_WORLD);
    makeHistogram(rank, size, aantalBins);


    MPI_Finalize();
    return 0;
}

vector<float> readFloats(const string &fname)
{
    vector<float> data;
    ifstream f(fname, std::ios::binary);

    f.seekg(0, ios_base::end);
    int pos = f.tellg();
    f.seekg(0, ios_base::beg);
    if (pos <= 0)
        throw runtime_error("Can't seek in file " + fname + " or file has zero length");

    if (pos % sizeof(float) != 0)
        throw runtime_error("File " + fname + " doesn't contain an integer number of float32 values");

    int num = pos/sizeof(float);
    data.resize(num);

    f.read(reinterpret_cast<char*>(data.data()), pos);
    if (f.gcount() != pos)
        throw runtime_error("Incomplete read: " + to_string(f.gcount()) + " vs " + to_string(pos));
    return data;
}

