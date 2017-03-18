#ifndef SSPP_COMMON_ABSTRACTCSRSOLVER_H_
#define SSPP_COMMON_ABSTRACTCSRSOLVER_H_

#include "CSR.h"
#include "Output.h"
#include "Definitions.h"
#include <vector>

namespace sspp {
  namespace tools {
    namespace solvers {
      class AbstractCSRSolver {
      public:
        AbstractCSRSolver() = default;
        virtual ~AbstractCSRSolver() = default;
        virtual representations::Output Solve(representations::CSR &csr, std::vector<FLOATING_TYPE> &b) = 0;
      };

    }
  }
}

#endif
