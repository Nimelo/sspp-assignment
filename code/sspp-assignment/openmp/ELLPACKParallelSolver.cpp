#include "ELLPACKParallelSolver.h"

representations::output::Output tools::solvers::parallel::ellpack::ELLPACKParallelSolver::solve(const representations::ellpack::ELLPACK & ellpack, FLOATING_TYPE * b, int threads)
{
	FLOATING_TYPE *x = new FLOATING_TYPE[ellpack.N];

	return representations::output::Output(ellpack.N, x);
}
