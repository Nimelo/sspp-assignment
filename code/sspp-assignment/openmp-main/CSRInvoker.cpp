#include "CSRInvoker.h"
#include "../common/ExecutionTimer.h"
#include "../openmp/CSRParallelSolver.h"
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
	std::string metadataFile = this->outputFile + META_EXTENSION;
	std::string outputFile = this->outputFile + OUTPUT_EXTENSION;
	result.save(metadataFile, outputFile);
}

tools::invokers::csr::CSRInvoker::CSRInvoker(std::string inputFile, std::string outputFile, int threads, int iterations)
	: inputFile(inputFile), outputFile(outputFile), threads(threads), iterations(iterations)
{
}

void tools::invokers::csr::CSRInvoker::invoke()
{
	representations::csr::CSR csr = loadCSR();
	FLOATING_TYPE *b = createVectorB(csr.N);
	
	representations::result::Result result;
	solvers::parallel::csr::CSRParallelSolver solver;
	representations::output::Output output;
	auto timer = tools::measurements::timers::ExecutionTimer();
	int numberOfThreads = threads;

	std::function<void()> solveCSRparallelRoutine = [&output, &solver, &csr, &b, &numberOfThreads]()
	{
		output = solver.solve(csr, b, numberOfThreads);
	};

	for (int i = 0; i < iterations; i++)
	{
		auto executionTime = timer.measure(solveCSRparallelRoutine);
		result.executionTimes.push_back(executionTime.count());
	}
	
	result.output = output;
	
	saveResult(result);

	delete[] b;
}
