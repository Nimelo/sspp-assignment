#ifndef __H_CSR_CUDA_SOLVER
#define __H_CSR_CUDA_SOLVER

#include "../common/CSR.h"
#include "../common/Output.h"
#include "../common/Definitions.h"
#include "../common/Output.h"

namespace tools
{
	namespace solvers
	{
		namespace cuda
		{
			namespace csr
			{
				class CSRCudaSolver
				{
				public:
					representations::output::Output solve(representations::csr::CSR &csr, FLOATING_TYPE *b);
				};
			}
		}		
	}
}

#endif