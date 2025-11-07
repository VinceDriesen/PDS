#include <cuda_runtime.h>
#include <iostream>

__global__ void ArrayKernel(int *blockArray, int *threadArray, int n) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < n) {
    blockArray[id] = blockIdx.x;
    threadArray[id] = threadIdx.x;
  }
}

int main(int argc, char *argv[]) {
  const int N = 500;
  const int blocks = 16;
  int threadsPerBlock = N / blocks;
  if (N % blocks != 0) {
    threadsPerBlock++;
  }

  int *blockArray;
  int *threadArray;

  cudaMallocManaged(&blockArray, sizeof(int) * N);
  cudaMallocManaged(&threadArray, sizeof(int) * N);

  ArrayKernel<<<blocks, threadsPerBlock>>>(blockArray, threadArray, N);

  cudaDeviceSynchronize(); // Ensure kernel is done

  for (int i = 0; i < N; ++i) {
    std::cout << "Element " << i << ": block " << blockArray[i] << ", thread "
              << threadArray[i] << '\n';
  }

  cudaFree(blockArray);
  cudaFree(threadArray);

  return 0;
}
