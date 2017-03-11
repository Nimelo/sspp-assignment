#include "ELLPACKParallelSolver.h"

representations::output::Output solvers::parallel::ellpack::ELLPACKParallelSolver::solve(const representations::ellpack::ELLPACK & ellpack, FLOATING_TYPE * b)
{
	FLOATING_TYPE *x = new FLOATING_TYPE[ellpack.N];

	return representations::output::Output(ellpack.N, x);
}
