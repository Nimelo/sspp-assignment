#include "ELLPACKCudaSolver.h"
#include <host_defines.h>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>

__global__ void ellpackKernel(unsigned rows, unsigned max_row_non_zeros, unsigned *JA, float *AS, float *b, float *x) {
  //TODO: Add ellpackKernel	
  int row = blockDim.x * blockIdx.x + threadIdx.x;

  if(row < rows) {
    float dot = 0;
    for(int i = 0; i < max_row_non_zeros; i++) {
      auto index = row * max_row_non_zeros + i;
      int col = JA[index];
      float val = AS[index];
      if(val != 0)
        dot += val * b[col];
    }
    x[row] += dot;
  }

}

sspp::common::Output<float> sspp::tools::solvers::ELLPACKCudaSolver::Solve(common::ELLPACK<float>& ellpack, std::vector<float>& b) {
  std::vector<float> x(ellpack.GetRows());
  //TODO: handle cudaStatus exceptions
  cudaError_t cudaStatus;
  float *device_x, *device_b, *device_as;
  unsigned *device_ja;

  cudaStatus = cudaMalloc(&device_x, sizeof(float) * x.size());
  cudaStatus = cudaMalloc(&device_b, sizeof(float) * b.size());
  cudaStatus = cudaMalloc(&device_as, sizeof(float) * ellpack.GetValues().size());
  cudaStatus = cudaMalloc(&device_ja, sizeof(unsigned) * ellpack.GetColumnIndices().size());

  cudaStatus = cudaMemcpy(device_x, &x[0], sizeof(float) * x.size(), cudaMemcpyHostToDevice);
  cudaStatus = cudaMemcpy(device_b, &b[0], sizeof(float) * b.size(), cudaMemcpyHostToDevice);
  cudaStatus = cudaMemcpy(device_as, &ellpack.GetValues()[0], sizeof(float) * ellpack.GetValues().size(), cudaMemcpyHostToDevice);
  cudaStatus = cudaMemcpy(device_ja, &ellpack.GetColumnIndices()[0], sizeof(unsigned) * ellpack.GetColumnIndices().size(), cudaMemcpyHostToDevice);
  unsigned int x_dimension = ellpack.GetRows() > 1024 ? 1024 : ellpack.GetRows();
  ellpackKernel << <x_dimension, 1 >> > (ellpack.GetRows(), ellpack.GetMaxRowNonZeros(), device_ja, device_as, device_b, device_x);

  cudaStatus = cudaMemcpy(&x[0], device_x, sizeof(float) * x.size(), cudaMemcpyDeviceToHost);

  cudaFree(device_x);
  cudaFree(device_b);
  cudaFree(device_as);
  cudaFree(device_ja);

  return common::Output<float>(x);
}
