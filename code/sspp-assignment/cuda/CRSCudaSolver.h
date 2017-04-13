#ifndef SSPP_CUDA_CSRCUDASOLVER_H_
#define SSPP_CUDA_CSRCUDASOLVER_H_

#include "../common/CRS.h"
#include "../common/Output.h"
#include "../common/AbstractCRSSolver.h"
#include "CUDAStopwatch.h"

namespace sspp {
  namespace cuda {
    namespace crs {
      void SetUpWrapper(int &device_id, unsigned long long &thread_blocks, unsigned long long &threads_per_block);

      template<typename VALUE_TYPE>
      void LoadCRS(common::CRS<VALUE_TYPE> & crs,
                   std::vector<VALUE_TYPE> & vector,
                   unsigned long long **d_row_start_indexes,
                   unsigned long long **d_column_indices,
                   VALUE_TYPE **d_values,
                   VALUE_TYPE **d_vector,
                   VALUE_TYPE **d_output);


      template<typename VALUE_TYPE>
      void ReleaseCRS(unsigned long long *d_row_start_indexes,
                      unsigned long long *d_column_indices,
                      VALUE_TYPE *d_values,
                      VALUE_TYPE *d_vector,
                      VALUE_TYPE *d_output);

      template <typename VALUE_TYPE>
      void SolveCRS(unsigned long long thread_blocks,
                    unsigned long long threads_per_block,
                    unsigned long long rows,
                    unsigned long long *d_row_start_indexes,
                    unsigned long long *d_column_indices,
                    VALUE_TYPE *d_values,
                    VALUE_TYPE *d_vector,
                    VALUE_TYPE *d_output);

      template<typename VALUE_TYPE>
      std::vector<VALUE_TYPE> GetResult(VALUE_TYPE *d_output,
                                        unsigned long long size);
    }
    template<typename VALUE_TYPE>
    class CRSCudaSolver : public common::AbstractCRSSolver<VALUE_TYPE> {
    public:
      common::Output<VALUE_TYPE> Solve(common::CRS<VALUE_TYPE>& crs, std::vector<VALUE_TYPE>& vector) {
        using namespace crs;
        int device_id_;
        unsigned long long thread_blocks, threads_per_block, *d_row_start_indexes, *d_column_indices;
        VALUE_TYPE *d_values, *d_vector, *d_output;
        SetUpWrapper(device_id_, thread_blocks, threads_per_block);

        LoadCRS(crs, vector, &d_row_start_indexes, &d_column_indices, &d_values, &d_vector, &d_output);
        CUDAStopwatch stopwatch;
        stopwatch.Start();
        SolveCRS(thread_blocks, threads_per_block, crs.GetRows(), d_row_start_indexes, d_column_indices, d_values, d_vector, d_output);
        stopwatch.Stop();
        std::vector<VALUE_TYPE> result = GetResult(d_output, crs.GetRows());
        ReleaseCRS(d_row_start_indexes, d_column_indices, d_values, d_vector, d_output);
        return common::Output<VALUE_TYPE>(result, stopwatch.GetElapsedSeconds());
      }
    };

  }
}

#endif
