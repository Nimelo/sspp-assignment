#include "ELLPACKCudaSolver.h"
#include <host_defines.h>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>

__global__ void ellpackKernel(INDEXING_TYPE rows, INDEXING_TYPE max_row_non_zeros, FLOATING_TYPE *JA, FLOATING_TYPE *AS, FLOATING_TYPE *b, FLOATING_TYPE *x) {
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

 /* cudaError_t cudaStatus;
  sspp::representations::ELLPACK *hEllpackPtr = &ellpack;
  sspp::representations::ELLPACK *dEllpackPtr;
  FLOATING_TYPE** h_AS = hEllpackPtr->AS;
  int** h_JA = hEllpackPtr->JA;

  int *d_JA;
  FLOATING_TYPE *d_AS, *d_B, *d_X;

  cudaMalloc(&d_JA, sizeof(int) * ellpack.rows_ * ellpack.max_row_non_zeros_);
  cudaMemcpy(d_JA, h_JA, sizeof(int) * ellpack.rows_ * ellpack.max_row_non_zeros_, cudaMemcpyHostToDevice);

  hEllpackPtr->JA = &d_JA;

  cudaMalloc(&d_AS, sizeof(FLOATING_TYPE) * ellpack.rows_ * ellpack.max_row_non_zeros_);
  cudaMemcpy(d_AS, h_AS, sizeof(FLOATING_TYPE) * ellpack.rows_ * ellpack.max_row_non_zeros_, cudaMemcpyHostToDevice);

  hEllpackPtr->AS = &d_AS;

  cudaMalloc(&d_B, sizeof(FLOATING_TYPE) * ellpack.rows_);
  cudaMemcpy(d_B, b, sizeof(FLOATING_TYPE) * ellpack.rows_ * ellpack.max_row_non_zeros_, cudaMemcpyHostToDevice);

  cudaMalloc(&d_X, sizeof(FLOATING_TYPE) * ellpack.rows_);

  ellpackKernel << <ellpack.rows_, 1 >> > (ellpack, d_B, d_X);

  cudaMemcpy(x, d_X, sizeof(FLOATING_TYPE)*ellpack.rows_, cudaMemcpyDeviceToHost);
*/
  return sspp::representations::Output(x);
}
