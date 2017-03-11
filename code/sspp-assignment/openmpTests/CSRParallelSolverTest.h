#pragma once

#include "ITest.h"
#include "../openmp/CSRParallelSolver.h"

class CSRParallelSolverTest : public ITest
{
protected:
	solvers::parallel::csr::CSRParallelSolver *csrParallerlSolver;
	virtual void SetUp()
	{
		csrParallerlSolver = new solvers::parallel::csr::CSRParallelSolver();
	}

	virtual void TearDown()
	{
		delete csrParallerlSolver;
	}
};