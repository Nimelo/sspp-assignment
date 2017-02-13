#ifndef __H_CSR_SOLVER
#define __H_CSR_SOLVER

#include "CSR.h"
#include "Output.h"
#include "Definitions.h"

namespace tools
{
	namespace solvers
	{
		namespace csr
		{
			class CSRSolver
			{
				public:
					representations::output::Output solve(representations::csr::CSR &csr, FLOATING_TYPE *vector);
			};
		}
	}
}

#endif