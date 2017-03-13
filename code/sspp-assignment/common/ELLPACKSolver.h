#ifndef __H_ELLPACK_SOLVER
#define __H_ELLPACK_SOLVER

#include "Definitions.h"
#include "ELLPACK.h"
#include "Output.h"
#include "AbstractELLPACKSolver.h"

namespace tools
{
	namespace solvers
	{
		namespace ellpack
		{
			class ELLPACKSolver : public AbstractELLPACKSolver
			{
			public:
				representations::output::Output solve(representations::ellpack::ELLPACK & ellpack, FLOATING_TYPE *b) override;
			};
		}
	}
}

#endif