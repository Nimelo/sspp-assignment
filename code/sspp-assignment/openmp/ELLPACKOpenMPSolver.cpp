#include "ELLPACKOpenMPSolver.h"

representations::output::Output tools::solvers::ellpack::ELLPACKOpenMPSolver::solve(representations::ellpack::ELLPACK & ellpack, FLOATING_TYPE * b)
{
	FLOATING_TYPE *x = new FLOATING_TYPE[ellpack.N];

	return representations::output::Output(ellpack.N, x);
}

void tools::solvers::ellpack::ELLPACKOpenMPSolver::setThreads(int threads)
{
	this->threads = threads;
}
