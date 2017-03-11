#ifndef __H_CSR_PARALLEL_SOLVER
#define __H_CSR_PARALLEL_SOLVER

#include "../common/CSR.h"
#include "../common/Definitions.h"
#include "../common/Output.h"

namespace solvers
{
	namespace parallel
	{
		namespace csr
		{
			class CSRParallelSolver
			{
				public:
					representations::output::Output solve(const representations::csr::CSR & csr, FLOATING_TYPE* b, int threads);
			};
		}
	}
}

#endif