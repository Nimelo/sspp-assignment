#ifndef SSPP_COMMON_CSRSOLVER_H_
#define SSPP_COMMON_CSRSOLVER_H_

#include "CSR.h"
#include "Output.h"
#include "Definitions.h"
#include "AbstractCSRSolver.h"

namespace sspp {
  namespace tools {
    namespace solvers {
      class CSRSolver : public AbstractCSRSolver {
      public:
        representations::Output solve(representations::CSR &csr, FLOATING_TYPE *b) override;
      };
    }
  }
}
#endif