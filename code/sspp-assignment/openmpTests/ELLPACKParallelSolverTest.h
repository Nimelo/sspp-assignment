#pragma once

#include "ITest.h"
#include "../openmp/ELLPACKOpenMPSolver.h"

class ELLPACKParallelSolverTest : public ITest
{
protected:
	tools::solvers::ellpack::ELLPACKOpenMPSolver *ellpackParallelSolver;
	virtual void SetUp()
	{
		ellpackParallelSolver = new tools::solvers::ellpack::ELLPACKOpenMPSolver();
	}

	virtual void TearDown()
	{
		delete ellpackParallelSolver;
	}
};