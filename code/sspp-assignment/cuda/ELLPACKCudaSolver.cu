#include "ELLPACKCudaSolver.h"
#include <host_defines.h>
#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include "helper_cuda.h"

namespace sspp {
  namespace cuda {
    namespace ellpack {
      template<typename VALUE_TYPE>
      __global__ void EllpackKernel(unsigned rows,
                                    unsigned max_row_non_zeros,
                                    unsigned *JA,
                                    VALUE_TYPE *AS,
                                    VALUE_TYPE *b,
                                    VALUE_TYPE *x) {
        int row = blockDim.x * blockIdx.x + threadIdx.x;

        if(row < rows) {
          VALUE_TYPE dot = VALUE_TYPE(0);
          for(unsigned i = 0; i < max_row_non_zeros; i++) {
            auto index = row * max_row_non_zeros + i;
            unsigned col = JA[index];
            VALUE_TYPE val = AS[index];
            if(val != 0)
              dot += val * b[col];
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
      void LoadELLPACK(common::ELLPACK<VALUE_TYPE>& ellpack,
                       std::vector<VALUE_TYPE>& vector,
                       unsigned** d_column_indices,
                       VALUE_TYPE** d_values,
                       VALUE_TYPE** d_vector,
                       VALUE_TYPE** d_output) {
        checkCudaErrors(cudaMalloc(d_vector, sizeof(VALUE_TYPE) * vector.size()));
        checkCudaErrors(cudaMalloc(d_output, sizeof(VALUE_TYPE) * ellpack.GetRows()));
        checkCudaErrors(cudaMalloc(d_values, sizeof(VALUE_TYPE) * ellpack.GetValues().size()));
        checkCudaErrors(cudaMalloc(d_column_indices, sizeof(unsigned) * ellpack.GetColumnIndices().size()));

        checkCudaErrors(cudaMemcpy(*d_vector, &vector[0], sizeof(VALUE_TYPE) * vector.size(), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemset(*d_output, 0, sizeof(VALUE_TYPE) * ellpack.GetRows()));
        checkCudaErrors(cudaMemcpy(*d_values, &ellpack.GetValues()[0], sizeof(VALUE_TYPE) * ellpack.GetValues().size(), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(*d_column_indices, &ellpack.GetColumnIndices()[0], sizeof(unsigned) * ellpack.GetColumnIndices().size(), cudaMemcpyHostToDevice));
      }

      template <typename VALUE_TYPE>
      void ReleaseELLPACK(unsigned* d_column_indices,
                          VALUE_TYPE* d_values,
                          VALUE_TYPE* d_vector,
                          VALUE_TYPE* d_output) {
        if(d_column_indices)
          checkCudaErrors(cudaFree(d_column_indices));
        if(d_values) {
          checkCudaErrors(cudaFree(d_values));
        }
        if(d_vector) {
          checkCudaErrors(cudaFree(d_vector));
        }
        if(d_output) {
          checkCudaErrors(cudaFree(d_output));
        }
      }

      template <typename VALUE_TYPE>
      void SolveELLPACK(unsigned thread_blocks,
                        unsigned threads_per_block,
                        unsigned rows,
                        unsigned max_row_non_zeros,
                        unsigned* d_column_indices,
                        VALUE_TYPE* d_values,
                        VALUE_TYPE* d_vector,
                        VALUE_TYPE* d_output) {
        auto blocks = std::ceil(static_cast<double>(rows) / threads_per_block);
        EllpackKernel << <blocks, threads_per_block >> > (rows, max_row_non_zeros, d_column_indices, d_values, d_vector, d_output);
        cudaDeviceSynchronize();
      }

      template <typename VALUE_TYPE>
      std::vector<VALUE_TYPE> GetResult(VALUE_TYPE* d_output, unsigned size) {
        std::vector<VALUE_TYPE> result(size);
        checkCudaErrors(cudaMemcpy(&result[0], d_output, sizeof(VALUE_TYPE) * size, cudaMemcpyDeviceToHost));
        return result;
      }
    }
    template class ELLPACKCudaSolver<float>;
    template class ELLPACKCudaSolver<double>;
  }
}