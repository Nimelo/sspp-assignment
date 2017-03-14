#include "CSROpenMPSolver.h"
#include <omp.h>

representations::output::Output tools::solvers::csr::CSROpenMPSolver::solve(representations::csr::CSR & csr, FLOATING_TYPE * b)
{
	FLOATING_TYPE *x = new FLOATING_TYPE[csr.M];

#pragma omp for
	for (auto i = 0; i < csr.M; i++)
		x[i] = 0.0;

#pragma omp parallel shared(csr, b, x)
	{
		int threads = omp_get_num_threads(),
			threadId = omp_get_thread_num();
		int lowerBoundary = csr.M * threadId / threads,
			upperBoundary = csr.M *(threadId + 1) / threads;

		for (auto i = lowerBoundary; i < upperBoundary; i++)
		{
			for (auto j = csr.IRP[i]; j < csr.IRP[i + 1]; j++)
			{
				x[i] += csr.AS[j] * b[csr.JA[j]];
			}
		}
	}
	//TODO: examine omp barriers
#pragma omp barrier

	return representations::output::Output(csr.M, x);
}

tools::solvers::csr::CSROpenMPSolver::CSROpenMPSolver(int threads)
	:threads(threads)
{
	omp_set_dynamic(0);
	omp_set_num_threads(threads);
}

void tools::solvers::csr::CSROpenMPSolver::setThreads(int threads)
{
	this->threads = threads;
	omp_set_dynamic(0);
	omp_set_num_threads(threads);
}
