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
        representations::Output Solve(representations::CSR &csr, std::vector<FLOATING_TYPE> &b) override;
      };
    }
  }
}
#endif