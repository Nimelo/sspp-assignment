#ifndef SSPP_COMMON_ABSTRACTCSRSOLVER_H_
#define SSPP_COMMON_ABSTRACTCSRSOLVER_H_

#include "CRS.h"
#include "Output.h"
#include <vector>

namespace sspp {
  namespace common {
    template<typename VALUE_TYPE>
    class AbstractCSRSolver {
    public:
      virtual ~AbstractCSRSolver() {};
      virtual Output<VALUE_TYPE> const & Solve(CRS<VALUE_TYPE> & crs, std::vector<VALUE_TYPE> & vector) = 0;
    };
  }
}

#endif
