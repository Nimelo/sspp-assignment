#include "CSRCudaSolver.h"
#include <host_defines.h>
#include <device_launch_parameters.h>

__global__ void csrKernel(sspp::representations::CSR & csr, FLOATING_TYPE* b, sspp::representations::Output & output) {
  int row = blockDim.x * blockIdx.x + threadIdx.x;
  if(row < csr.M) {
    FLOATING_TYPE dot = 0;
    int rowStart = csr.IRP[row],
      rowEnd = csr.IRP[row + 1];

    for(int i = rowStart; i < rowEnd; i++) {
      dot += csr.AS[i] * b[csr.JA[i]];
    }

    output.Values[row] += dot;
  }

}

sspp::representations::Output sspp::tools::solvers::CSRCudaSolver::solve(sspp::representations::CSR & csr, FLOATING_TYPE * b) {
  FLOATING_TYPE *x = new FLOATING_TYPE[csr.M];

  return representations::Output(csr.M, x);
}
