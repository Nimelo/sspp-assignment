#ifndef __H_ELLPACK_PARALLEL_SOLVER
#define __H_ELLPACK_PARALLEL_SOLVER

#include "../common/ELLPACK.h"
#include "../common/Output.h"
#include "../common/AbstractELLPACKSolver.h"
#include <omp.h>

namespace sspp {
  namespace openmp {
    template<typename VALUE_TYPE>
    class ELLPACKOpenMPSolver : public common::AbstractELLPACKSolver<VALUE_TYPE> {
    public:
      static void SetThreads(int threads) {
        omp_set_dynamic(0);
        omp_set_num_threads(threads);
      }

      common::Output<VALUE_TYPE> Solve(common::ELLPACK<VALUE_TYPE> & ellpack, std::vector<VALUE_TYPE> & b) {
        std::vector<VALUE_TYPE> x(ellpack.GetRows());

#pragma omp parallel shared(ellpack, b, x)
        for(auto i = 0; i < ellpack.GetRows(); i++) {
          VALUE_TYPE tmp = 0;
          for(auto j = 0; j < ellpack.GetMaxRowNonZeros(); j++) {
            auto index = ellpack.CalculateIndex(i, j);
            tmp += ellpack.GetValues()[index] * b[ellpack.GetColumnIndices()[index]];
          }

          x[i] = tmp;
        }
        return common::Output<VALUE_TYPE>(x);
      };
    };
  }
}
#endif