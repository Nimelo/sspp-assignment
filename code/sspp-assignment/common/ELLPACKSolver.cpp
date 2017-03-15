#include "ELLPACKSolver.h"

sspp::representations::Output sspp::tools::solvers::ELLPACKSolver::solve(sspp::representations::ELLPACK & ellpack, FLOATING_TYPE * b)
{
	FLOATING_TYPE *x = new FLOATING_TYPE[ellpack.M];

	for (auto i = 0; i < ellpack.M; i++)
	{
		FLOATING_TYPE tmp = 0;
		for (auto j = 0; j < ellpack.MAXNZ; j++)
		{
			tmp += ellpack.AS[i][j] * b[ellpack.JA[i][j]];
		}

		x[i] = tmp;
	}

	return sspp::representations::Output(ellpack.M, x);
}
