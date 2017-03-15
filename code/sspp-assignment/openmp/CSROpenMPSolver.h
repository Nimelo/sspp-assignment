#ifndef __H_CSR_PARALLEL_SOLVER
#define __H_CSR_PARALLEL_SOLVER

#include "../common/CSR.h"
#include "../common/Definitions.h"
#include "../common/Output.h"
#include "../common/AbstractCSRSolver.h"

namespace sspp {
  namespace tools {
    namespace solvers {
      class CSROpenMPSolver : public sspp::tools::solvers::AbstractCSRSolver {
      protected:
        int threads = 1;
      public:
        CSROpenMPSolver(int threads);
        CSROpenMPSolver() = default;
        void setThreads(int threads);
        sspp::representations::Output solve(sspp::representations::CSR & csr, FLOATING_TYPE* b) override;
      };
    }
  }
}
#endif