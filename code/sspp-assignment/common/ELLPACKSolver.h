#ifndef SSPP_COMMON_ELLPACKSOLVER_H_
#define SSPP_COMMON_ELLPACKSOLVER_H_

#include "ELLPACK.h"
#include "Output.h"
#include "AbstractELLPACKSolver.h"

namespace sspp {
  namespace common {
    template<typename VALUE_TYPE>
    class ELLPACKSolver : public AbstractELLPACKSolver<VALUE_TYPE> {
    public:
      Output<VALUE_TYPE> Solve(ELLPACK<VALUE_TYPE> & ellpack, std::vector<VALUE_TYPE> & vector) {
        std::vector<VALUE_TYPE> x(ellpack.GetRows());

        for(unsigned i = 0; i < ellpack.GetRows(); ++i) {
          VALUE_TYPE tmp = 0;
          for(unsigned j = 0; j < ellpack.GetMaxRowNonZeros(); ++j) {
            auto index = ellpack.CalculateIndex(i, j);
            tmp += ellpack.GetValues()[index] * vector[ellpack.GetColumnIndices()[index]];
          }

          x[i] = tmp;
        }

        return Output<VALUE_TYPE>(x);
      }
    };
  }
}

#endif