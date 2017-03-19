#include "CSRCudaSolver.h"
#include <host_defines.h>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>

__global__ void csrKernel(INDEXING_TYPE rows, INDEXING_TYPE *irp, INDEXING_TYPE *ja, FLOATING_TYPE *as, FLOATING_TYPE* b, FLOATING_TYPE *x) {
  int row = blockDim.x * blockIdx.x + threadIdx.x;
  if(row < rows) {
    FLOATING_TYPE dot = 0;
    int rowStart = irp[row],
      rowEnd = irp[row + 1];

    for(int i = rowStart; i < rowEnd; i++) {
      dot += as[i] * b[ja[i]];
    }

    x[row] += dot;
  }
}

sspp::representations::Output sspp::tools::solvers::CSRCudaSolver::Solve(sspp::representations::CSR & csr, std::vector<FLOATING_TYPE> & b) {
  std::vector<FLOATING_TYPE> x(csr.GetRows());
  cudaError_t cudaStatus;
  INDEXING_TYPE *device_irp, *device_ja;
  FLOATING_TYPE *device_as, *device_b, *device_x;
  //TODO: handle cudaStatus exceptions
  cudaStatus = cudaMalloc(&device_irp, sizeof(INDEXING_TYPE) * csr.GetIRP().size());
  cudaStatus = cudaMalloc(&device_ja, sizeof(INDEXING_TYPE) * csr.GetJA().size());
  cudaStatus = cudaMalloc(&device_as, sizeof(FLOATING_TYPE) * csr.GetAS().size());
  cudaStatus = cudaMalloc(&device_b, sizeof(FLOATING_TYPE) * b.size());
  cudaStatus = cudaMalloc(&device_x, sizeof(FLOATING_TYPE) * x.size());

  cudaStatus = cudaMemcpy(device_irp, &csr.GetIRP()[0], sizeof(INDEXING_TYPE) * csr.GetIRP().size(), cudaMemcpyHostToDevice);
  cudaStatus = cudaMemcpy(device_ja, &csr.GetJA()[0], sizeof(INDEXING_TYPE) * csr.GetJA().size(), cudaMemcpyHostToDevice);
  cudaStatus = cudaMemcpy(device_as, &csr.GetAS()[0], sizeof(FLOATING_TYPE) * csr.GetAS().size(), cudaMemcpyHostToDevice);
  cudaStatus = cudaMemcpy(device_b, &b[0], sizeof(FLOATING_TYPE) * b.size(), cudaMemcpyHostToDevice);
  cudaStatus = cudaMemcpy(device_x, &x[0], sizeof(FLOATING_TYPE) * x.size(), cudaMemcpyHostToDevice);
  unsigned int x_dimension = csr.GetRows() > 1024 ? 1024 : csr.GetRows();
  csrKernel << <x_dimension, 1 >> > (csr.GetRows(), device_irp, device_ja, device_as, device_b, device_x);

  cudaStatus = cudaMemcpy(&x[0], device_x, sizeof(FLOATING_TYPE) * x.size(), cudaMemcpyDeviceToHost);

  cudaFree(device_irp);
  cudaFree(device_ja);
  cudaFree(device_as);
  cudaFree(device_b);
  cudaFree(device_x);

  return representations::Output(x);
}
