#ifndef SSPP_COMMON_ABSTRACTCSRSOLVER_H_
#define SSPP_COMMON_ABSTRACTCSRSOLVER_H_

#include "CSR.h"
#include "Output.h"
#include "Definitions.h"

namespace sspp {
  namespace tools {
    namespace solvers {
      class AbstractCSRSolver {
      public:
        virtual ~AbstractCSRSolver() = default;
        virtual representations::Output solve(representations::CSR &csr, FLOATING_TYPE *b) = 0;
      };

    }
  }
}

#endif
