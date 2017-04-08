#ifndef SSPP_OPENMP_ELLPACKOPENMPSOLVER_H_
#define SSPP_OPENMP_ELLPACKOPENMPSOLVER_H_
#include <omp.h>

#include "../common/ELLPACK.h"
#include "../common/Output.h"
#include "../common/AbstractELLPACKSolver.h"
#include "OpenMPStopwatch.h"

namespace sspp {
  namespace openmp {
    template<typename VALUE_TYPE>
    class ELLPACKOpenMPSolver : public common::AbstractELLPACKSolver<VALUE_TYPE> {
    public:
      static void SetThreads(int threads) {
        omp_set_dynamic(0);
        omp_set_num_threads(threads);
      }

      virtual common::Output<VALUE_TYPE> Solve(common::ELLPACK<VALUE_TYPE> & ellpack, std::vector<VALUE_TYPE> & b) {
        static OpenMPStopwatch stopwatch_;
        std::vector<VALUE_TYPE> x(ellpack.GetRows());

        stopwatch_.Start();
#pragma omp parallel shared(ellpack, b, x)
        {
          int threads = omp_get_num_threads(),
            threadId = omp_get_thread_num();
          int lowerBoundary = ellpack.GetRows() * threadId / threads,
            upperBoundary = ellpack.GetRows() *(threadId + 1) / threads;

          for(unsigned i = lowerBoundary; i < upperBoundary; i++) {
            VALUE_TYPE tmp = 0;
            for(unsigned j = 0; j < ellpack.GetMaxRowNonZeros(); j++) {
              auto index = ellpack.CalculateIndex(i, j);
              tmp += ellpack.GetValues()[index] * b[ellpack.GetColumnIndices()[index]];
            }

            x[i] = tmp;
          }
        }
        stopwatch_.Stop();
        return common::Output<VALUE_TYPE>(x, stopwatch_.GetElapsedSeconds());
      };
    };
  }
}
#endif