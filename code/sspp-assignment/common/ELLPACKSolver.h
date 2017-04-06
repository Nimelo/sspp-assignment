#ifndef SSPP_COMMON_ELLPACKSOLVER_H_
#define SSPP_COMMON_ELLPACKSOLVER_H_

#include "../common/ELLPACK.h"
#include "../common/Output.h"
#include "../common/AbstractELLPACKSolver.h"
#include <chrono>
#include "ChronoStopwatch.h"

namespace sspp {
  namespace common {
    template<typename VALUE_TYPE>
    class ELLPACKSolver : public common::AbstractELLPACKSolver<VALUE_TYPE> {
    public:
      common::Output<VALUE_TYPE> Solve(common::ELLPACK<VALUE_TYPE> & ellpack, std::vector<VALUE_TYPE> & vector) {
        std::vector<VALUE_TYPE> x(ellpack.GetRows());
        static ChronoStopwatch stopwatch_;
       
        stopwatch_.Start();
        for(unsigned i = 0; i < ellpack.GetRows(); ++i) {
          VALUE_TYPE tmp = 0;
          for(unsigned j = 0; j < ellpack.GetMaxRowNonZeros(); ++j) {
            auto index = ellpack.CalculateIndex(i, j);
            tmp += ellpack.GetValues()[index] * vector[ellpack.GetColumnIndices()[index]];
          }

          x[i] = tmp;
        }
        stopwatch_.Stop();
        return common::Output<VALUE_TYPE>(x, stopwatch_.GetElapsedSeconds());
      }
    };
  }
}

#endif