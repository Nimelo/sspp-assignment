#ifndef __H_ELLPACK_CUDA_SOLVER
#define __H_ELLPACK_CUDA_SOLVER

#include "../common/ELLPACK.h"
#include "../common/Output.h"
#include "../common/Definitions.h"
#include "../common/AbstractELLPACKSolver.h"

namespace tools
{
	namespace solvers
	{
		namespace ellpack
		{
			class ELLPACKCudaSolver :public AbstractELLPACKSolver
			{
			public:
				representations::output::Output solve(representations::ellpack::ELLPACK &ellpack, FLOATING_TYPE *b) override;
			};
		}
	}
}

#endif