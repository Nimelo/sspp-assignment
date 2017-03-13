#ifndef __H_ABSTRACT_ELLPACK_SOLVER
#define __H_ABSTRACT_ELLPACK_SOLVER

#include "ELLPACK.h"
#include "Output.h"
#include "Definitions.h"

namespace tools
{
	namespace solvers
	{
		namespace ellpack
		{
			class AbstractELLPACKSolver
			{
			public:
				virtual ~AbstractELLPACKSolver() = default;
				virtual representations::output::Output solve(representations::ellpack::ELLPACK &ellpack, FLOATING_TYPE *b) = 0;
			};
		}
	}
}

#endif
