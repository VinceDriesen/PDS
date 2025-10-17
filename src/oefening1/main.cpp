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

void histogram(int aantalBins, const vector<float> &histValues, float minValue, float maxValue) {
    std::vector<int> histogram(aantalBins, 0);

    float binWidth = (maxValue - minValue) / aantalBins;

    for (float v : histValues) {
        int bin = static_cast<int>((v - minValue) / binWidth);
        if (bin == aantalBins) bin--;
        histogram[bin]++;
    }

    std::cout << "Histogram (" << aantalBins << " bins):" << std::endl;

    for (int i = 0; i < aantalBins; ++i) {
        float binStart = minValue + i * binWidth;
        float binEnd = binStart + binWidth;

        std::cout << "[" << binStart << ", " << binEnd << "]: " << histogram[i] << std::endl;
    }
}



int main()
{
    int aantalBins;
    std::cout << "Oefening 1" << std::endl;
    std::cout << "Geef aantal bins: ";
    std::cin >> aantalBins;
    std::cout << "Aantal bins: " << aantalBins << std::endl;
    vector<float> histValues = readFloats("resources/histvalues.dat");
    
    auto [minValue, maxValue] = findMinMax(histValues);
    std::cout << "Max value: " << maxValue << std::endl;
    std::cout << "Min value: " << minValue << std::endl;
    histogram(aantalBins, histValues, minValue, maxValue);

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

