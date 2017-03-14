#pragma once

#include "ITest.h"
#include "../cuda/ELLPACKCudaSolver.h"

class CudaELLPACKSolverTest : public ITest
{
protected:
	tools::solvers::ellpack::ELLPACKCudaSolver *csrParallelSolver;
	virtual void SetUp()
	{
		csrParallelSolver = new tools::solvers::ellpack::ELLPACKCudaSolver();
	}

	virtual void TearDown()
	{
		delete csrParallelSolver;
	}
};