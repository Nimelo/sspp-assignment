#ifndef SSPP_CUDA_ELLPACKCUDASOLVER_H_
#define SSPP_CUDA_ELLPACKCUDASOLVER_H_

#include "../common/ELLPACK.h"
#include "../common/Output.h"
#include "../common/AbstractELLPACKSolver.h"
#include "CUDAStopwatch.h"

namespace sspp {
  namespace cuda {
    namespace ellpack {
      void SetUpWrapper(int &device_id, unsigned &thread_blocks, unsigned &threads_per_block);

      template<typename VALUE_TYPE>
      void LoadELLPACK(common::ELLPACK<VALUE_TYPE> & ellpack,
                       std::vector<VALUE_TYPE> & vector,
                       unsigned **d_column_indices,
                       VALUE_TYPE **d_values,
                       VALUE_TYPE **d_vector,
                       VALUE_TYPE **d_output);

      template<typename VALUE_TYPE>
      void ReleaseELLPACK(unsigned *d_column_indices,
                          VALUE_TYPE *d_values,
                          VALUE_TYPE *d_vector,
                          VALUE_TYPE *d_output);

      template <typename VALUE_TYPE>
      void SolveELLPACK(unsigned thread_blocks,
                        unsigned threads_per_block,
                        unsigned rows,
                        unsigned max_row_non_zeros,
                        unsigned *d_column_indices,
                        VALUE_TYPE *d_values,
                        VALUE_TYPE *d_vector,
                        VALUE_TYPE *d_output);

      template<typename VALUE_TYPE>
      std::vector<VALUE_TYPE> GetResult(VALUE_TYPE *d_output, unsigned size);
    }

    template<typename VALUE_TYPE>
    class ELLPACKCudaSolver :public common::AbstractELLPACKSolver<VALUE_TYPE> {
    public:
      common::Output<VALUE_TYPE> Solve(common::ELLPACK<VALUE_TYPE>& ellpack, std::vector<VALUE_TYPE>& vector) {
        using namespace ellpack;
        int device_id;
        unsigned thread_blocks, threads_per_block, *d_column_indices;
        VALUE_TYPE *d_values, *d_vector, *d_output;
        SetUpWrapper(device_id, thread_blocks, threads_per_block);
        LoadELLPACK(ellpack, vector, &d_column_indices, &d_values, &d_vector, &d_output);
        CUDAStopwatch stopwatch;
        stopwatch.Start();
        SolveELLPACK(thread_blocks, threads_per_block, ellpack.GetRows(), ellpack.GetMaxRowNonZeros(), d_column_indices, d_values, d_vector, d_output);
        stopwatch.Stop();
        ReleaseELLPACK(d_column_indices, d_values, d_vector, d_output);
        return common::Output<VALUE_TYPE>(GetResult(d_output, ellpack.GetRows()), stopwatch.GetElapsedSeconds());
      }
    };
  }
}

#endif
