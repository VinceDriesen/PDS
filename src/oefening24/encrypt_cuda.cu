#include "parameters.h"
#include <cuda.h>
#include <stdint.h>
#include <vector>

__global__ void encryptKernel(const uint8_t *plaintext, uint8_t *output,
                              const uint8_t *roundkeys, int numRounds,
                              const uint8_t *substitutionTable,
                              const uint16_t *permutationTable, bool decrypt) {
  extern __shared__ uint8_t sharedMem[];
  uint8_t *firstPart = sharedMem;
  uint8_t *secondPart = sharedMem + 512;

  int blockId = blockIdx.x;
  int threadId = threadIdx.x;

  if (threadId < 512) {
    const int offset = blockId * 1024;
    firstPart[threadId] = plaintext[offset + threadId];
    secondPart[threadId] = plaintext[offset + 512 + threadId];
  }

  __syncthreads();

  int startKey, stopKey, step;
  if (!decrypt) {
    startKey = 0;
    stopKey = numRounds - 1;
    step = 1;
  } else {
    startKey = numRounds - 1;
    stopKey = 0;
    step = -1;
  }

  for (int k = startKey; (step > 0) ? (k <= stopKey) : (k >= stopKey);
       k += step) {
    const uint8_t *roundKey = &roundkeys[k * 512];
    __syncthreads();

    uint8_t *temp = sharedMem + 1024;

    // Dit is de hele runF functie
    if (threadId < 512) {
      uint8_t val = secondPart[threadId] ^ roundKey[threadId];
      val = substitutionTable[val];
      temp[permutationTable[threadId]] = val;
    }
    __syncthreads();

    if (threadId < 512) {
      firstPart[threadId] ^= temp[threadId];
    }
    __syncthreads();

    if (k != stopKey) {
      if (threadId < 512) {
        uint8_t tempVal = firstPart[threadId];
        firstPart[threadId] = secondPart[threadId];
        secondPart[threadId] = tempVal;
      }
    }
    __syncthreads();
  }

  if (threadId < 512) {
    const int offset = blockId * 1024;
    output[offset + threadId] = firstPart[threadId];
    output[offset + 512 + threadId] = secondPart[threadId];
  }
}
/*

Aangezien ik het mezelf leuk vind om het altijd moeilijk te maken, kiez ik om
alle rounds uit te voeren in 1 kernel call ipv elke round in een aparte kernel
call te doen.

Dus de de opsplitsing van de eerste for loop (die om first en second part te
krijgen) moeten we al in de kernel doen. We beginnen bij de kernel parameters:
Deze moet eigenlijk alles krijgen...
inclusief de round nummers, en een susbtitution en permutation table.


De kernel begint dan met het maken van first en second part. Aangezien shared
memory, hoort dit daarbij.

Als we dat hebben kunnen we beginnen met de stappen:
stap 1: de bytes inladen in shared memory. Niet vergeten te synchronizeren na
het inladen.
stap2: round range bepalen (dit is de if else in het begin van de
pseudo code basically)
stap 3: de effectieve round uitvoeren (de for loop in de
pseudo code)
stap4: output terugschrijven naar global memory.
*/

std::vector<uint8_t> encrypt_cuda(std::vector<uint8_t> plaintext,
                                  std::vector<std::vector<uint8_t>> roundkeys,
                                  bool decrypt) {
  std::vector<uint8_t> output(plaintext.size(), 0);
  const size_t numBlocks = plaintext.size() / 1024;
  const int threadsPerBlock = 512;
  const int sharedMemSize = 3 * 512;

  std::vector<uint8_t> flatKeys;
  for (auto &rk : roundkeys)
    flatKeys.insert(flatKeys.end(), rk.begin(), rk.end());
  const int numRounds = roundkeys.size();

  uint8_t *d_plaintext = nullptr;
  uint8_t *d_output = nullptr;
  uint8_t *d_roundkeys = nullptr;
  uint8_t *d_subTable = nullptr;
  uint16_t *d_permTable = nullptr;

  cudaMalloc(&d_plaintext, plaintext.size() * sizeof(uint8_t));
  cudaMalloc(&d_output, output.size() * sizeof(uint8_t));
  cudaMalloc(&d_roundkeys, flatKeys.size() * sizeof(uint8_t));
  cudaMalloc(&d_subTable, 256 * sizeof(uint8_t));
  cudaMalloc(&d_permTable, 512 * sizeof(uint16_t));

  cudaMemcpy(d_plaintext, plaintext.data(), plaintext.size(),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_roundkeys, flatKeys.data(), flatKeys.size(),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_subTable, substitutionTable, 256, cudaMemcpyHostToDevice);
  cudaMemcpy(d_permTable, permutationTable, 512 * sizeof(uint16_t),
             cudaMemcpyHostToDevice);

  encryptKernel<<<numBlocks, threadsPerBlock, sharedMemSize>>>(
      d_plaintext, d_output, d_roundkeys, numRounds, d_subTable, d_permTable,
      decrypt);

  cudaDeviceSynchronize();

  cudaMemcpy(output.data(), d_output, output.size(), cudaMemcpyDeviceToHost);

  cudaFree(d_plaintext);
  cudaFree(d_output);
  cudaFree(d_roundkeys);
  cudaFree(d_subTable);
  cudaFree(d_permTable);

  return output;
}
