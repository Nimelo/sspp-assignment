#include "CSRSolver.h"

representations::output::Output tools::solvers::csr::CSRSolver::solve(representations::csr::CSR &csr, FLOATING_TYPE * b)
{
	FLOATING_TYPE *x = new FLOATING_TYPE[csr.M];

	for (auto i = 0; i < csr.M; i++)
	{
		FLOATING_TYPE tmp = 0;

		for (auto j = csr.IRP[i]; j < csr.IRP[i + 1]; j++)
		{
			tmp += csr.AS[j] * b[csr.JA[j]];
		}

		x[i] = tmp;
	}

	return representations::output::Output(csr.M, x);
}
