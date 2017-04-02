#ifndef SSPP_OPENMP_CRSOPENMPSOLVER_H_
#define SSPP_OPENMP_CRSOPENMPSOLVER_H_

#include "../common/Output.h"
#include "../common/AbstractCRSSolver.h"
#include <omp.h>
#include <chrono>

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
        std::vector<VALUE_TYPE> x(crs.GetRows());

        double t1 = omp_get_wtime();
        std::chrono::steady_clock::time_point t11 = std::chrono::high_resolution_clock::now();
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
        double t2 = omp_get_wtime();
        std::chrono::steady_clock::time_point t22 = std::chrono::high_resolution_clock::now();
        return common::Output<VALUE_TYPE>(x, std::chrono::duration_cast<std::chrono::microseconds>(t22 - t11).count() / 1000000.0 );//(t2 - t1));
      };
    };
  }
}
#endif