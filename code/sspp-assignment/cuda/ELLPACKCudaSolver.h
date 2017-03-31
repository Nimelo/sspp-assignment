#ifndef SSPP_CUDA_ELLPACKCUDASOLVER_H_
#define SSPP_CUDA_ELLPACKCUDASOLVER_H_

#include "../common/ELLPACK.h"
#include "../common/Output.h"
#include "../common/AbstractELLPACKSolver.h"

namespace sspp {
  namespace tools {
    namespace solvers {
      class ELLPACKCudaSolver :public common::AbstractELLPACKSolver<float> {
      public:
        common::Output<float> Solve(common::ELLPACK<float>& ellpack, std::vector<float>& vector);
      };
    }
  }
}

#endif
