#ifndef __H_ELLPACK_CUDA_SOLVER
#define __H_ELLPACK_CUDA_SOLVER

#include "../common/ELLPACK.h"
#include "../common/Output.h"
#include "../common/Definitions.h"

namespace tools
{
	namespace solvers
	{
		namespace cuda
		{
			namespace csr
			{
				class ELLPACKCudaSolver
				{
				public:
					representations::output::Output solve(representations::ellpack::ELLPACK &ellpack, FLOATING_TYPE *b);
				};
			}
		}
	}
}

#endif