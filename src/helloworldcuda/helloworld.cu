#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

__global__ void Hello() {
  // printf works, not cout!!
  printf("Hello from thread %d\n", threadIdx.x);
}

void checkDevice() {
  int numDevs = 0;
  cudaGetDeviceCount(&numDevs);
  if (numDevs != 1)
    throw std::runtime_error("Expecting one CUDA device, but got " +
                             std::to_string(numDevs));
}

int main(int argc, char *argv[]) {
  checkDevice();
  if (argc != 2)
    throw std::runtime_error("Specify number of threads");
  int numThreads = std::stoi(argv[1]);
  Hello<<<1, numThreads>>>();
  cudaDeviceSynchronize();
  cudaError_t r = cudaGetLastError();
  if (r != cudaSuccess)
    throw std::runtime_error("Kernel failed, code is " + std::to_string(r) +
                             ", message: " + cudaGetErrorString(r));
  return 0;
}
