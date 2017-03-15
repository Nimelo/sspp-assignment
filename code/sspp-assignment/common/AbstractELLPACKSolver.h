#ifndef SSPP_COMMON_ABSTRACTELLPACKSOLVER_H
#define SSPP_COMMON_ABSTRACTELLPACKSOLVER_H

#include "ELLPACK.h"
#include "Output.h"
#include "Definitions.h"
namespace sspp {
  namespace tools {
    namespace solvers {
      class AbstractELLPACKSolver {
      public:
        virtual ~AbstractELLPACKSolver() = default;
        virtual representations::Output solve(representations::ELLPACK &ellpack, FLOATING_TYPE *b) = 0;
      };
    }
  }
}

#endif
