#ifndef SSPP_CUDA_CSRCUDASOLVER_H_
#define SSPP_CUDA_CSRCUDASOLVER_H_

#include "../common/CSR.h"
#include "../common/Output.h"
#include "../common/Definitions.h"
#include "../common/Output.h"
#include "../common/AbstractCSRSolver.h"

namespace sspp {
  namespace tools {
    namespace solvers {
      class CSRCudaSolver : public AbstractCSRSolver {
      public:
        representations::Output solve(representations::CSR &csr, FLOATING_TYPE *b) override;
      };
    }
  }
}

#endif
