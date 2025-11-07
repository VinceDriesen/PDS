#include "image2d.h"
#include <cuda.h>
#include <iostream>
#include <stdio.h>
#include <string>

#define BLOCKSIZE 16

__global__ void CUDAKernel(int iterations, float xmin, float xmax, float ymin,
                           float ymax, float *pOutput, int outputW,
                           int outputH) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= outputW || y >= outputH) {
    return;
  }

  float real = xmin + (xmax - xmin) * x / (outputW - 1);
  float imag = ymin + (ymax - ymin) * y / (outputH - 1);

  float zReal = 0.0f;
  float zImag = 0.0f;
  int Q = 0;

  while (Q < iterations && (zReal * zReal + zImag * zImag <= 4.0f)) {
    float tempReal = zReal * zReal - zImag * zImag + real;
    zImag = 2.0f * zReal * zImag + imag;
    zReal = tempReal;
    Q++;
  }

  if (zReal * zReal + zImag * zImag <= 4.0f) {
    pOutput[y * outputW + x] = 0.0f; // Inside the set
  } else {
    pOutput[y * outputW + x] = 255.0f * sqrtf((float)Q / iterations);
  }
}

// If an error occurs, return false and set a description in 'errStr'
bool cudaFractal(int iterations, float xmin, float xmax, float ymin, float ymax,
                 Image2D &output, std::string &errStr) {
  // We'll use an image of 512 pixels wide
  int ho = 512;
  int wo = ho * 3 / 2;
  output.resize(wo, ho);

  // And divide this in a number of blocks
  size_t xBlockSize = BLOCKSIZE;
  size_t yBlockSize = BLOCKSIZE;
  size_t numXBlocks = (wo / xBlockSize) + (((wo % xBlockSize) != 0) ? 1 : 0);
  size_t numYBlocks = (ho / yBlockSize) + (((ho % yBlockSize) != 0) ? 1 : 0);

  cudaError_t err;

  float *pDevOutput;

  cudaMallocManaged(&pDevOutput, wo * ho * sizeof(float));

  cudaEvent_t startEvt, stopEvt; // We'll use cuda events to time everything
  cudaEventCreate(&startEvt);
  cudaEventCreate(&stopEvt);

  cudaEventRecord(startEvt);

  dim3 grid(numXBlocks, numYBlocks);
  dim3 block(xBlockSize, yBlockSize);
  CUDAKernel<<<grid, block>>>(iterations, xmin, xmax, ymin, ymax, pDevOutput,
                              wo, ho);

  cudaDeviceSynchronize();

  cudaEventRecord(stopEvt);

  err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::cout << "CUDA convolution kernel execution error code: " << err
              << std::endl;
  }

  float *imgPtr = output.getBufferPointer();
  for (int i = 0; i < wo * ho; i++) {

    imgPtr[i] = pDevOutput[i];
  }

  cudaFree(pDevOutput);

  float elapsed;
  cudaEventElapsedTime(&elapsed, startEvt, stopEvt);

  std::cout << "CUDA time elapsed: " << elapsed << " milliseconds" << std::endl;

  cudaEventDestroy(startEvt);
  cudaEventDestroy(stopEvt);

  return true;
}
