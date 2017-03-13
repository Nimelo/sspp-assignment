#ifndef __H_CSR_PARALLEL_SOLVER
#define __H_CSR_PARALLEL_SOLVER

#include "../common/CSR.h"
#include "../common/Definitions.h"
#include "../common/Output.h"
#include "../common/AbstractCSRSolver.h"

namespace tools
{
	namespace solvers
	{
		namespace csr
		{
			class CSROpenMPSolver : public AbstractCSRSolver
			{
			protected:
				int threads = 1;
			public:
				CSROpenMPSolver(int threads);
				CSROpenMPSolver() = default;
				void setThreads(int threads);
				representations::output::Output solve(representations::csr::CSR & csr, FLOATING_TYPE* b) override;
			};
		}
	}
}
#endif