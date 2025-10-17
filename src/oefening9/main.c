#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define ARRAY_SIZE 1048576

void fill_array(float *arr, int size, float value) {
    for (int i = 0; i < size; i++) {
        arr[i] = value;
    }
}

void process0(float *arr1, float *arr2) {
    MPI_Ssend(arr1, ARRAY_SIZE, MPI_FLOAT, 1, 0, MPI_COMM_WORLD);

    MPI_Recv(arr2, ARRAY_SIZE, MPI_FLOAT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    printf("I am process 0 and I have received b(0) = %.2f\n", arr2[0]);
}

void process1(float *arr1, float *arr2) {
    MPI_Recv(arr2, ARRAY_SIZE, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    MPI_Ssend(arr1, ARRAY_SIZE, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);

    printf("I am process 1 and I have received b(0) = %.2f\n", arr2[0]);
}

void allocate_arrays(float **arr1, float **arr2, int rank) {
    *arr1 = (float *)malloc(ARRAY_SIZE * sizeof(float));
    *arr2 = (float *)malloc(ARRAY_SIZE * sizeof(float));
    if (*arr1 == NULL || *arr2 == NULL) {
        fprintf(stderr, "Error allocating memory\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    fill_array(*arr1, ARRAY_SIZE, rank);
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 2) {
        if (rank == 0) {
            fprintf(stderr, "Het is enkel voor 2 MPI processen gemaakt!\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    float *arr1, *arr2;
    allocate_arrays(&arr1, &arr2, rank);
    
    if (rank == 0) {
        process0(arr1, arr2);
    } else if (rank == 1) {
        process1(arr1, arr2);
    }


    free(arr1);
    free(arr2);
    MPI_Finalize();
    return 0;
}
