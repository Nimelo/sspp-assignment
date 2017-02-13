#include "CSRSolver.h"

representations::output::Output tools::solvers::csr::CSRSolver::solve(representations::csr::CSR &csr, FLOATING_TYPE * x)
{
	int N = csr.getM();
	FLOATING_TYPE *b = new FLOATING_TYPE[N];

	for (int i = 0; i < csr.getM(); i++)
	{
		FLOATING_TYPE tmp = 0;

		for (int j = csr.getIRP()[i]; j < csr.getIRP()[i+1]-1; j++)
		{
			tmp += csr.getAS()[j] * x[csr.getJA()[j]];
		}

		b[i] = tmp;
	}

	return representations::output::Output(N, b);
}
