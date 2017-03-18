#ifndef SSPP_COMMON_ELLPACKSOLVER_H_
#define SSPP_COMMON_ELLPACKSOLVER_H_

#include "Definitions.h"
#include "ELLPACK.h"
#include "Output.h"
#include "AbstractELLPACKSolver.h"

namespace sspp {
  namespace tools {
    namespace solvers {
      class ELLPACKSolver : public AbstractELLPACKSolver {
      public:
        representations::Output Solve(representations::ELLPACK & ellpack, std::vector<FLOATING_TYPE> &b) override;
      };
    }
  }
}

#endif