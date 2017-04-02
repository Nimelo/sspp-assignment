#include "CRSCudaSolver.h"
#include <host_defines.h>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include "helper_cuda.h"

namespace sspp {
  namespace cuda {
    namespace crs {
      template<typename VALUE_TYPE>
      __global__ void CrsKernel(unsigned rows, 
                                unsigned *irp, 
                                unsigned *ja,
                                VALUE_TYPE *as,
                                VALUE_TYPE *b,
                                VALUE_TYPE *x) {
        int row = blockDim.x * blockIdx.x + threadIdx.x;
        if(row < rows) {
          VALUE_TYPE dot = 0;
          unsigned rowStart = irp[row],
            rowEnd = irp[row + 1];

          for(unsigned i = rowStart; i < rowEnd; i++) {
            dot += as[i] * b[ja[i]];
          }

          x[row] += dot;
        }
      }

      void SetUpWrapper(int& device_id,
                        unsigned& thread_blocks,
                        unsigned& threads_per_block) {
        device_id = gpuGetMaxGflopsDeviceId();
        cudaSetDevice(device_id);
        cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
        cudaDeviceProp device_properties;
        checkCudaErrors(cudaGetDevice(&device_id));
        checkCudaErrors(cudaGetDeviceProperties(&device_properties, device_id));
        threads_per_block = device_properties.maxThreadsPerBlock;
        thread_blocks = device_properties.multiProcessorCount * device_properties.maxThreadsPerMultiProcessor / threads_per_block;
      }

      template <typename VALUE_TYPE>
      void LoadCRS(common::CRS<VALUE_TYPE>& crs,
                   std::vector<VALUE_TYPE>& vector,
                   unsigned** d_row_start_indexes,
                   unsigned** d_column_indices,
                   VALUE_TYPE** d_values,
                   VALUE_TYPE** d_vector,
                   VALUE_TYPE** d_output) {
        checkCudaErrors(cudaMalloc(d_row_start_indexes, sizeof(unsigned) * crs.GetRowStartIndexes().size()));
        checkCudaErrors(cudaMalloc(d_column_indices, sizeof(unsigned) * crs.GetColumnIndices().size()));
        checkCudaErrors(cudaMalloc(d_values, sizeof(VALUE_TYPE) * crs.GetValues().size()));
        checkCudaErrors(cudaMalloc(d_output, sizeof(VALUE_TYPE) * crs.GetRows()));
        checkCudaErrors(cudaMalloc(d_vector, sizeof(VALUE_TYPE) * vector.size()));

        checkCudaErrors(cudaMemcpy(*d_row_start_indexes, &crs.GetRowStartIndexes()[0], sizeof(unsigned) * crs.GetRowStartIndexes().size(), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(*d_column_indices, &crs.GetColumnIndices()[0], sizeof(unsigned) * crs.GetColumnIndices().size(), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(*d_values, &crs.GetValues()[0], sizeof(VALUE_TYPE) * crs.GetValues().size(), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemset(d_output, 0, sizeof(VALUE_TYPE) * crs.GetRows()));
        checkCudaErrors(cudaMemcpy(*d_vector, &vector[0], sizeof(VALUE_TYPE) * vector.size(), cudaMemcpyHostToDevice));
      }

      template <typename VALUE_TYPE>
      void ReleaseCRS(unsigned* d_row_start_indexes, 
                      unsigned* d_column_indices,
                      VALUE_TYPE* d_values,
                      VALUE_TYPE* d_vector,
                      VALUE_TYPE* d_output) {
        if(d_row_start_indexes)
          checkCudaErrors(cudaFree(d_row_start_indexes));
        if(d_column_indices)
          checkCudaErrors(cudaFree(d_column_indices));
        if(d_values)
          checkCudaErrors(cudaFree(d_values));
        if(d_vector)
          checkCudaErrors(cudaFree(d_vector));
        if(d_output)
          checkCudaErrors(cudaFree(d_output));
      }

      template <typename VALUE_TYPE>
      void SolveCRS(unsigned thread_blocks,
                    unsigned threads_per_block,
                    unsigned rows,
                    unsigned* d_row_start_indexes,
                    unsigned* d_column_indices,
                    VALUE_TYPE* d_values,
                    VALUE_TYPE* d_vector,
                    VALUE_TYPE* d_output) {
        CrsKernel<<<thread_blocks, threads_per_block>>>(rows, d_row_start_indexes, d_column_indices, d_values, d_vector, d_output);
        cudaDeviceSynchronize();
      }

      template <typename VALUE_TYPE>
      std::vector<VALUE_TYPE> GetResult(VALUE_TYPE* d_output, unsigned size) {
        std::vector<VALUE_TYPE> result(size);
        checkCudaErrors(cudaMemcpy(&result[0], d_output, sizeof(VALUE_TYPE) * size, cudaMemcpyDeviceToHost));
        return result;
      }
    }
    template class CRSCudaSolver<float>;
    template class CRSCudaSolver<double>;
  }
}