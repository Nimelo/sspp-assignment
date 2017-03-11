#ifndef __H_CSR_PARALLEL_SOLVER
#define __H_CSR_PARALLEL_SOLVER

#include "../common/ELLPACK.h"
#include "../common/Definitions.h"
#include "../common/Output.h"

namespace solvers
{
	namespace parallel
	{
		namespace ellpack
		{
			class EllpackParallelSolver
			{
			public:
				representations::output::Output solve(const representations::ellpack::ELLPACK & ellpack, FLOATING_TYPE* b);
			};
		}
	}
}

#endif