#ifndef SSPP_CUDA_ELLPACKCUDASOLVER_H_
#define SSPP_CUDA_ELLPACKCUDASOLVER_H_

#include "../common/ELLPACK.h"
#include "../common/Output.h"
#include "../common/Definitions.h"
#include "../common/AbstractELLPACKSolver.h"

namespace sspp {
  namespace tools {
    namespace solvers {
      class ELLPACKCudaSolver :public sspp::tools::solvers::AbstractELLPACKSolver {
      public:
        sspp::representations::Output Solve(sspp::representations::ELLPACK &ellpack, std::vector<FLOATING_TYPE> & b) override;
      };
    }
  }
}

#endif
