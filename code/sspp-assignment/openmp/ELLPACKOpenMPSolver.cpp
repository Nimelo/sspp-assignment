#include "ELLPACKOpenMPSolver.h"
#include <omp.h>

representations::output::Output tools::solvers::ellpack::ELLPACKOpenMPSolver::solve(representations::ellpack::ELLPACK & ellpack, FLOATING_TYPE * b)
{
	FLOATING_TYPE *x = new FLOATING_TYPE[ellpack.M];
#pragma omp for
	for (auto i = 0; i < ellpack.M; i++)
		x[i] = 0.0;

#pragma omp parallel shared(ellpack, b, x)
	for (auto i = 0; i < ellpack.M; i++)
	{
		FLOATING_TYPE tmp = 0;
		for (auto j = 0; j < ellpack.MAXNZ; j++)
		{
			tmp += ellpack.AS[i][j] * b[ellpack.JA[i][j]];
		}

		x[i] = tmp;
	}

	//TODO: examine omp barriers
#pragma omp barrier
	return representations::output::Output(ellpack.M, x);
}

tools::solvers::ellpack::ELLPACKOpenMPSolver::ELLPACKOpenMPSolver(int threads)
	:threads(threads)
{
	omp_set_dynamic(0);
	omp_set_num_threads(threads);
}

void tools::solvers::ellpack::ELLPACKOpenMPSolver::setThreads(int threads)
{
	this->threads = threads;
	omp_set_dynamic(0);
	omp_set_num_threads(threads);
}
