#pragma once

#include "ITest.h"
#include "../openmp/ELLPACKParallelSolver.h"

class ELLPACKParallelSolverTest : public ITest
{
protected:
	tools::solvers::parallel::ellpack::ELLPACKParallelSolver *ellpackParallelSolver;
	virtual void SetUp()
	{
		ellpackParallelSolver = new tools::solvers::parallel::ellpack::ELLPACKParallelSolver();
	}

	virtual void TearDown()
	{
		delete ellpackParallelSolver;
	}
};