#ifndef SSPP_CUDA_CSRCUDASOLVER_H_
#define SSPP_CUDA_CSRCUDASOLVER_H_

#include "../common/CRS.h"
#include "../common/Output.h"
#include "../common/Output.h"
#include "../common/AbstractCRSSolver.h"

namespace sspp {
  namespace tools {
    namespace solvers {
      class CRSCudaSolver : public common::AbstractCRSSolver<float> {
      public:
        common::Output<float> Solve(common::CRS<float>& crs, std::vector<float>& vector);
      };
    }
  }
}

#endif
