#include "CRSCudaSolver.h"
#include <host_defines.h>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>

__global__ void crsKernel(unsigned rows, unsigned *irp, unsigned *ja, float *as, float* b, float *x) {
  int row = blockDim.x * blockIdx.x + threadIdx.x;
  if(row < rows) {
    float dot = 0;
    int rowStart = irp[row],
      rowEnd = irp[row + 1];

    for(int i = rowStart; i < rowEnd; i++) {
      dot += as[i] * b[ja[i]];
    }

    x[row] += dot;
  }
}

sspp::common::Output<float> sspp::tools::solvers::CRSCudaSolver::Solve(common::CRS<float>& crs, std::vector<float>& b) {
  std::vector<float> x(crs.GetRows());
  cudaError_t cudaStatus;
  unsigned *device_irp, *device_ja;
  float *device_as, *device_b, *device_x;
  //TODO: handle cudaStatus exceptions
  cudaStatus = cudaMalloc(&device_irp, sizeof(unsigned) * crs.GetRowStartIndexes().size());
  cudaStatus = cudaMalloc(&device_ja, sizeof(unsigned) * crs.GetColumnIndices().size());
  cudaStatus = cudaMalloc(&device_as, sizeof(float) * crs.GetValues().size());
  cudaStatus = cudaMalloc(&device_b, sizeof(float) * b.size());
  cudaStatus = cudaMalloc(&device_x, sizeof(float) * x.size());

  cudaStatus = cudaMemcpy(device_irp, &crs.GetRowStartIndexes()[0], sizeof(unsigned) * crs.GetRowStartIndexes().size(), cudaMemcpyHostToDevice);
  cudaStatus = cudaMemcpy(device_ja, &crs.GetColumnIndices()[0], sizeof(unsigned) * crs.GetColumnIndices().size(), cudaMemcpyHostToDevice);
  cudaStatus = cudaMemcpy(device_as, &crs.GetValues()[0], sizeof(float) * crs.GetValues().size(), cudaMemcpyHostToDevice);
  cudaStatus = cudaMemcpy(device_b, &b[0], sizeof(float) * b.size(), cudaMemcpyHostToDevice);
  cudaStatus = cudaMemcpy(device_x, &x[0], sizeof(float) * x.size(), cudaMemcpyHostToDevice);
  unsigned int x_dimension = crs.GetRows() > 1024 ? 1024 : crs.GetRows();
  crsKernel << <x_dimension, 1 >> > (crs.GetRows(), device_irp, device_ja, device_as, device_b, device_x);

  cudaStatus = cudaMemcpy(&x[0], device_x, sizeof(float) * x.size(), cudaMemcpyDeviceToHost);

  cudaFree(device_irp);
  cudaFree(device_ja);
  cudaFree(device_as);
  cudaFree(device_b);
  cudaFree(device_x);

  return common::Output<float>(x);
}
