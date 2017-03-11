#include "CSRParallelSolver.h"

representations::output::Output solvers::parallel::csr::CSRParallelSolver::solve(const representations::csr::CSR & csr, FLOATING_TYPE * b, int threads)
{
	FLOATING_TYPE *x = new FLOATING_TYPE[csr.N];

	return representations::output::Output(csr.N, x);
}
