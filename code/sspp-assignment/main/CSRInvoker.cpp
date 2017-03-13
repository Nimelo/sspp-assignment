#include "CSRInvoker.h"
#include "../common/ExecutionTimer.h"
#include "../openmp/CSROpenMPSolver.h"
#include "../common/CSRSolver.h"
#include "../common/Result.h"
#include "../common/Definitions.h"

#include <fstream>

representations::csr::CSR tools::invokers::csr::CSRInvoker::loadCSR()
{
	std::fstream is;
	is.open(inputFile, std::fstream::in);
	representations::csr::CSR csr;
	is >> csr;
	is.close();

	return csr;
}

FLOATING_TYPE * tools::invokers::csr::CSRInvoker::createVectorB(int n)
{
	FLOATING_TYPE *b = new FLOATING_TYPE[n];
	for (int i = 0; i < n; i++)
		b[i] = 1;

	return b;
}

void tools::invokers::csr::CSRInvoker::saveResult(representations::result::Result & result)
{
	std::string outputFile = this->outputFile + DASH_CSR + OUTPUT_EXTENSION;
	std::fstream fs;
	fs.open(outputFile, std::fstream::out | std::fstream::trunc);
	fs << result;
	fs.close();
}

tools::invokers::csr::CSRInvoker::CSRInvoker(std::string inputFile, std::string outputFile, int iterationsParallel, int iterationsSerial)
	: inputFile(inputFile), outputFile(outputFile), iterationsParallel(iterationsParallel), iterationsSerial(iterationsSerial)
{
}

void tools::invokers::csr::CSRInvoker::invoke(solvers::csr::AbstractCSRSolver & parallelSolver)
{
	representations::csr::CSR csr = loadCSR();
	FLOATING_TYPE *b = createVectorB(csr.N);
	
	representations::result::Result result;
	tools::solvers::csr::CSRSolver serialSolver;

	representations::output::Output output;
	auto timer = tools::measurements::timers::ExecutionTimer();

	std::function<void()> solveCSRSerialRoutine = [&output, &serialSolver, &csr, &b]()
	{
		output = serialSolver.solve(csr, b);
	};

	std::function<void()> solveCSRparallelRoutine = [&output, &parallelSolver, &csr, &b]()
	{
		output = parallelSolver.solve(csr, b);
	};

	for (int i = 0; i < iterationsSerial; i++)
	{
		auto executionTime = timer.measure(solveCSRSerialRoutine);
		result.serialResult.executionTimes.push_back(executionTime.count());
	}
	result.serialResult.output = output;

	for (int i = 0; i < iterationsParallel; i++)
	{
		auto executionTime = timer.measure(solveCSRparallelRoutine);
		result.parallelResult.executionTimes.push_back(executionTime.count());
	}
	result.parallelResult.output = output;
	
	saveResult(result);

	delete[] b;
}
