#pragma once

#include "ITest.h"
#include "../openmp/CSROpenMPSolver.h"

class CSRParallelSolverTest : public ITest
{
protected:
  sspp::tools::solvers::CSROpenMPSolver *csrParallelSolver;
	virtual void SetUp()
	{
		csrParallelSolver = new sspp::tools::solvers::CSROpenMPSolver();
	}

	virtual void TearDown()
	{
		delete csrParallelSolver;
	}
};