#ifndef __H_ABSTRACT_CSR_SOLVER
#define __H_ABSTRACT_CSR_SOLVER

#include "CSR.h"
#include "Output.h"
#include "Definitions.h"

namespace tools
{
	namespace solvers
	{
		namespace csr
		{
			class AbstractCSRSolver
			{
			public:
				virtual ~AbstractCSRSolver() = default;
				virtual representations::output::Output solve(representations::csr::CSR &csr, FLOATING_TYPE *b) = 0;
			};
		}
	}
}

#endif
