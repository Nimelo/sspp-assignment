#pragma once

#include "ITest.h"
#include "../cuda/CSRCudaSolver.h"

class CudaCSRSolverTest : public ITest
{
protected:
	tools::solvers::csr::CSRCudaSolver *csrParallelSolver;
	virtual void SetUp()
	{
		csrParallelSolver = new tools::solvers::csr::CSRCudaSolver();
	}

	virtual void TearDown()
	{
		delete csrParallelSolver;
	}
};