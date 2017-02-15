#include "CSRSolver.h"

representations::output::Output tools::solvers::csr::CSRSolver::solve(representations::csr::CSR &csr, FLOATING_TYPE * x)
{
	FLOATING_TYPE *b = new FLOATING_TYPE[csr.M];

	for (int i = 0; i < csr.M; i++)
	{
		FLOATING_TYPE tmp = 0;

		for (int j = csr.IRP[i]; j < csr.IRP[i+1]; j++)
		{
			tmp += csr.AS[j] * x[csr.JA[j]];
		}

		b[i] = tmp;
	}

	return representations::output::Output(csr.M, b);
}
