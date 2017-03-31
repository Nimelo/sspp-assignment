#ifndef SSPP_COMMON_ELLPACKSOLVER_H_
#define SSPP_COMMON_ELLPACKSOLVER_H_

#include "../common/ELLPACK.h"
#include "../common/Output.h"
#include "../common/AbstractELLPACKSolver.h"

namespace sspp {
  namespace serial {
    template<typename VALUE_TYPE>
    class ELLPACKSolver : public common::AbstractELLPACKSolver<VALUE_TYPE> {
    public:
      common::Output<VALUE_TYPE> Solve(common::ELLPACK<VALUE_TYPE> & ellpack, std::vector<VALUE_TYPE> & vector) {
        std::vector<VALUE_TYPE> x(ellpack.GetRows());

        auto t1 = std::chrono::high_resolution_clock::now();
        for(unsigned i = 0; i < ellpack.GetRows(); ++i) {
          VALUE_TYPE tmp = 0;
          for(unsigned j = 0; j < ellpack.GetMaxRowNonZeros(); ++j) {
            auto index = ellpack.CalculateIndex(i, j);
            tmp += ellpack.GetValues()[index] * vector[ellpack.GetColumnIndices()[index]];
          }

          x[i] = tmp;
        }
        auto t2 = std::chrono::high_resolution_clock::now();
        return common::Output<VALUE_TYPE>(x, std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count());
      }
    };
  }
}

#endif