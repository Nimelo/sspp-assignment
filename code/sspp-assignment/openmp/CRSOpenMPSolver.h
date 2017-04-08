#ifndef SSPP_OPENMP_CRSOPENMPSOLVER_H_
#define SSPP_OPENMP_CRSOPENMPSOLVER_H_

#include <omp.h>

#include "../common/Output.h"
#include "../common/AbstractCRSSolver.h"

#include <chrono>
#include "OpenMPStopwatch.h"

namespace sspp {
  namespace openmp {
    template<typename VALUE_TYPE>
    class CRSOpenMPSolver : public common::AbstractCRSSolver<VALUE_TYPE> {
    public:
      static void SetThreads(int threads) {
        omp_set_dynamic(0);
        omp_set_num_threads(threads);
      }

      common::Output<VALUE_TYPE> Solve(common::CRS<VALUE_TYPE>& crs, std::vector<VALUE_TYPE>& vector) {
        static OpenMPStopwatch stopwatch_;
        std::vector<VALUE_TYPE> x(crs.GetRows());

        stopwatch_.Start();
#pragma omp parallel shared(crs, vector, x)
        {
          int threads = omp_get_num_threads(),
            threadId = omp_get_thread_num();
          int lowerBoundary = crs.GetRows() * threadId / threads,
            upperBoundary = crs.GetRows() *(threadId + 1) / threads;

          //#pragma ivdep
          for(auto i = lowerBoundary; i < upperBoundary; i++) {
            for(auto j = crs.GetRowStartIndexes()[i]; j < crs.GetRowStartIndexes()[i + 1]; j++) {
              x[i] += crs.GetValues()[j] * vector[crs.GetColumnIndices()[j]];
            }
          }
        }
        
        stopwatch_.Stop();
        return common::Output<VALUE_TYPE>(x, stopwatch_.GetElapsedSeconds());
      };
    };
  }
}
#endif