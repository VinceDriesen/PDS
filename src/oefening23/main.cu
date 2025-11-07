#include <cuda_runtime.h>
#include <iostream>

__global__ void reverseArrayKernel(int *input, int *output, const int N) {
  __shared__ int sharedData[32];

  int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
  int localIdx = threadIdx.x;

  // Kopieer van global naar shared memory
  sharedData[localIdx] = input[globalIdx];
  __syncthreads();

  // Schrijf omgekeerd terug naar global memory
  int reversedIdx = (N - 1) - globalIdx;
  output[reversedIdx] = sharedData[localIdx];
}

int main() {
  const int N = 512;
  const int BLOCK_SIZE = 32;
  const int GRID_SIZE = 16;

  int inputArray[N];
  int outputArray[N];

  for (int i = 0; i < N; i++) {
    inputArray[i] = i;
  }

  int *d_inputArray, *d_outputArray;
  cudaMalloc((void **)&d_inputArray, N * sizeof(int));
  cudaMalloc((void **)&d_outputArray, N * sizeof(int));

  cudaMemcpy(d_inputArray, inputArray, N * sizeof(int), cudaMemcpyHostToDevice);

  // Hier moet ik de kernel aanroepen om de array om te keren
  reverseArrayKernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_inputArray, d_outputArray, N);
  cudaDeviceSynchronize();

  cudaMemcpy(outputArray, d_outputArray, N * sizeof(int),
             cudaMemcpyDeviceToHost);

  for (int i = 0; i < N; i++) {
    std::cout << outputArray[i] << " ";
    if ((i + 1) % 32 == 0)
      std::cout << "\n";
  }

  cudaFree(d_inputArray);
  cudaFree(d_outputArray);

  return 0;
}
