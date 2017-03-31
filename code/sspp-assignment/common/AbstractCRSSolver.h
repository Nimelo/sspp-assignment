#ifndef SSPP_COMMON_ABSTRACTCSRSOLVER_H_
#define SSPP_COMMON_ABSTRACTCSRSOLVER_H_

#include "CRS.h"
#include "Output.h"
#include <vector>

namespace sspp {
  namespace common {
    template<typename VALUE_TYPE>
    class AbstractCRSSolver {
    public:
      virtual ~AbstractCRSSolver() {};
      virtual Output<VALUE_TYPE> Solve(CRS<VALUE_TYPE> & crs, std::vector<VALUE_TYPE> & vector) = 0;
    };
  }
}

#endif
