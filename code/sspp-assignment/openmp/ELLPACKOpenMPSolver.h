#ifndef __H_CSR_PARALLEL_SOLVER
#define __H_CSR_PARALLEL_SOLVER

#include "../common/ELLPACK.h"
#include "../common/Definitions.h"
#include "../common/Output.h"
#include "../common/AbstractELLPACKSolver.h"

namespace tools
{
	namespace solvers
	{
		namespace ellpack
		{
			class ELLPACKOpenMPSolver : public AbstractELLPACKSolver
			{
			protected:
				int threads = 1;
			public:
				void setThreads(int threads);
				representations::output::Output solve(representations::ellpack::ELLPACK & ellpack, FLOATING_TYPE* b) override;
			};
		}
	}
}
#endif