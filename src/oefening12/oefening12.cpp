#include <mpi.h>
#include <iostream>
#include <vector>
#include <fstream>

void process0(std::vector<int>& arr1, std::vector<int>& arr2, int array_size) {
    MPI_Send(arr1.data(), array_size, MPI_INT, 1, 0, MPI_COMM_WORLD);
    MPI_Recv(arr2.data(), array_size, MPI_INT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

void process1(std::vector<int>& arr1, std::vector<int>& arr2, int array_size) {
    MPI_Recv(arr2.data(), array_size, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Send(arr1.data(), array_size, MPI_INT, 0, 0, MPI_COMM_WORLD);
}

double measureAvrRTTTime(int rank, int bufferSize, int repeats) {
    double avgTime = 0;
    std::vector<int> data1(bufferSize / sizeof(int));
    std::vector<int> data2(bufferSize / sizeof(int));

    for (int i = 0; i < repeats; i++) {
        double start = MPI_Wtime();
        if (rank == 0)
            process0(data1, data2, data1.size());
        else
            process1(data1, data2, data1.size());
        double end = MPI_Wtime();
        avgTime += (end - start);
    }

    return avgTime / repeats;
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 2) {
        if (rank == 0)
            fprintf(stderr, "Het programma werkt enkel met 2 MPI-processen!\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    std::ofstream out("output_rank" + std::to_string(rank) + ".txt");

    for (int n = 0; n <= 10; n++) {
        int bufferSize = 1 << n; // 2^n bytes
        double avgTime = measureAvrRTTTime(rank, bufferSize, 1'000'000); // 10^6
        out << "BufferSize: " << bufferSize
            << " bytes, Avg RTT: " << avgTime * 1e6 << " µs" << std::endl;
    }

    out.close();
    MPI_Finalize();
    return 0;
}

