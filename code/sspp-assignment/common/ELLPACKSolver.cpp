#include "ELLPACKSolver.h"

representations::output::Output tools::solvers::ellpack::ELLPACKSolver::solve(representations::ellpack::ELLPACK & ellpack, FLOATING_TYPE * b)
{
	FLOATING_TYPE *x = new FLOATING_TYPE[ellpack.M];

	for (int i = 0; i < ellpack.M; i++)
	{
		FLOATING_TYPE tmp = 0;
		for (int j = 0; j < ellpack.MAXNZ; j++)
		{
			tmp += ellpack.AS[i][j] * b[ellpack.JA[i][j]];
		}

		x[i] = tmp;
	}

	return representations::output::Output(ellpack.M, x);
}
