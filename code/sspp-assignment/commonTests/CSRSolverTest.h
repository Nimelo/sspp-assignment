#pragma once
#include "ITest.h"
#include "CSRSolver.h"
class CSRSolverTest : public ITest
{
protected:
	tools::solvers::csr::CSRSolver *csrSolver;
	virtual void SetUp()
	{
		csrSolver = new tools::solvers::csr::CSRSolver();
	}

	virtual void TearDown()
	{
		delete csrSolver;
	}
};