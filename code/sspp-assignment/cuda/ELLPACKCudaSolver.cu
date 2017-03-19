#include "ELLPACKCudaSolver.h"
#include <host_defines.h>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>

__global__ void ellpackKernel(INDEXING_TYPE rows, INDEXING_TYPE max_row_non_zeros, INDEXING_TYPE *JA, FLOATING_TYPE *AS, FLOATING_TYPE *b, FLOATING_TYPE *x) {
  //TODO: Add ellpackKernel	
  int row = blockDim.x * blockIdx.x + threadIdx.x;

  if(row < rows) {
    FLOATING_TYPE dot = 0;
    for(int i = 0; i < max_row_non_zeros; i++) {
      auto index = row * max_row_non_zeros + i;
      int col = JA[index];
      FLOATING_TYPE val = AS[index];
      if(val != 0)
        dot += val * b[col];
    }
    x[row] += dot;
  }

}

sspp::representations::Output sspp::tools::solvers::ELLPACKCudaSolver::Solve(sspp::representations::ELLPACK & ellpack, std::vector<FLOATING_TYPE> & b) {
  std::vector<FLOATING_TYPE> x(ellpack.GetRows());
  //TODO: handle cudaStatus exceptions
  cudaError_t cudaStatus;
  FLOATING_TYPE *device_x, *device_b, *device_as;
  INDEXING_TYPE *device_ja;

  cudaStatus = cudaMalloc(&device_x, sizeof(FLOATING_TYPE) * x.size());
  cudaStatus = cudaMalloc(&device_b, sizeof(FLOATING_TYPE) * b.size());
  cudaStatus = cudaMalloc(&device_as, sizeof(FLOATING_TYPE) * ellpack.GetAS().size());
  cudaStatus = cudaMalloc(&device_ja, sizeof(INDEXING_TYPE) * ellpack.GetJA().size());

  cudaStatus = cudaMemcpy(device_x, &x[0], sizeof(FLOATING_TYPE) * x.size(), cudaMemcpyHostToDevice);
  cudaStatus = cudaMemcpy(device_b, &b[0], sizeof(FLOATING_TYPE) * b.size(), cudaMemcpyHostToDevice);
  cudaStatus = cudaMemcpy(device_as, &ellpack.GetAS()[0], sizeof(FLOATING_TYPE) * ellpack.GetAS().size(), cudaMemcpyHostToDevice);
  cudaStatus = cudaMemcpy(device_ja, &ellpack.GetJA()[0], sizeof(INDEXING_TYPE) * ellpack.GetJA().size(), cudaMemcpyHostToDevice);

  ellpackKernel << <ellpack.GetRows(), 1 >> > (ellpack.GetRows(), ellpack.GetMaxRowNonZeros(), device_ja, device_as, device_b, device_x);

  cudaStatus = cudaMemcpy(&x[0], device_x, sizeof(FLOATING_TYPE) * x.size(), cudaMemcpyDeviceToHost);

  cudaFree(device_x);
  cudaFree(device_b);
  cudaFree(device_as);
  cudaFree(device_ja);

  return sspp::representations::Output(x);
}
