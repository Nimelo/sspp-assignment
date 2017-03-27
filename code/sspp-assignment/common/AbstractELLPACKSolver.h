#ifndef SSPP_COMMON_ABSTRACTELLPACKSOLVER_H
#define SSPP_COMMON_ABSTRACTELLPACKSOLVER_H

#include "ELLPACK.h"
#include "Output.h"

namespace sspp {
  namespace common {
    template<typename VALUE_TYPE>
    class AbstractELLPACKSolver {
    public:
      virtual ~AbstractELLPACKSolver() {};
      virtual Output<VALUE_TYPE> const & Solve(ELLPACK<VALUE_TYPE> & ellpack, std::vector<VALUE_TYPE> & vector) = 0;
    };
  }
}

#endif
