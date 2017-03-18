#include "CSRCudaSolver.h"
#include <host_defines.h>
#include <device_launch_parameters.h>

__global__ void csrKernel(INDEXING_TYPE rows, INDEXING_TYPE *IRP, INDEXING_TYPE *JA, INDEXING_TYPE *AS, FLOATING_TYPE* b, FLOATING_TYPE *x) {
  int row = blockDim.x * blockIdx.x + threadIdx.x;
  if(row < rows) {
    FLOATING_TYPE dot = 0;
    int rowStart = IRP[row],
      rowEnd = IRP[row + 1];

    for(int i = rowStart; i < rowEnd; i++) {
      dot += AS[i] * b[JA[i]];
    }

    x[row] += dot;
  }
}

sspp::representations::Output sspp::tools::solvers::CSRCudaSolver::Solve(sspp::representations::CSR & csr, std::vector<FLOATING_TYPE> & b) {
  std::vector<FLOATING_TYPE> x(csr.GetRows());

  return representations::Output(x);
}
