#pragma once
#include "ITest.h"
#include "EllpackSolver.h"
class ELLPACKSolverTest : public ITest
{
protected:
	tools::solvers::ellpack::ELLPACKSolver *ellpackSolver;
	virtual void SetUp()
	{
		ellpackSolver = new tools::solvers::ellpack::ELLPACKSolver();
	}

	virtual void TearDown()
	{
		delete ellpackSolver;
	}
};
