#ifndef __H_CSR_CUDA_SOLVER
#define __H_CSR_CUDA_SOLVER

#include "../common/CSR.h"
#include "../common/Output.h"
#include "../common/Definitions.h"
#include "../common/Output.h"
#include "../common/AbstractCSRSolver.h"

namespace tools
{
	namespace solvers
	{
		namespace csr
		{
			class CSRCudaSolver : public AbstractCSRSolver
			{
			public:
				representations::output::Output solve(representations::csr::CSR &csr, FLOATING_TYPE *b) override;
			};
		}
	}
}

#endif