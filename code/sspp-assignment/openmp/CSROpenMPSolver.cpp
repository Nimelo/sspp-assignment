#include "CSROpenMPSolver.h"

representations::output::Output tools::solvers::csr::CSROpenMPSolver::solve(representations::csr::CSR & csr, FLOATING_TYPE * b)
{
	FLOATING_TYPE *x = new FLOATING_TYPE[csr.N];

	return representations::output::Output(csr.N, x);
}

tools::solvers::csr::CSROpenMPSolver::CSROpenMPSolver(int threads)
	:threads(threads)
{
}

void tools::solvers::csr::CSROpenMPSolver::setThreads(int threads)
{
	this->threads = threads;
}
