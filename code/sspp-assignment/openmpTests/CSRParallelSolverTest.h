#pragma once

#include "ITest.h"
#include "../openmp/CSROpenMPSolver.h"

class CSRParallelSolverTest : public ITest
{
protected:
	tools::solvers::csr::CSROpenMPSolver *csrParallelSolver;
	virtual void SetUp()
	{
		csrParallelSolver = new tools::solvers::csr::CSROpenMPSolver();
	}

	virtual void TearDown()
	{
		delete csrParallelSolver;
	}
};