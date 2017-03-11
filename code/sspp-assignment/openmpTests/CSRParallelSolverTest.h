#pragma once

#include "ITest.h"
#include "../openmp/CSRParallelSolver.h"

class CSRParallelSolverTest : public ITest
{
protected:
	solvers::parallel::csr::CSRParallelSolver *csrParallelSolver;
	virtual void SetUp()
	{
		csrParallelSolver = new solvers::parallel::csr::CSRParallelSolver();
	}

	virtual void TearDown()
	{
		delete csrParallelSolver;
	}
};