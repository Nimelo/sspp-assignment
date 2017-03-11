#include "ELLPACKInvoker.h"
#include "../common/ExecutionTimer.h"
#include "../openmp/ELLPACKParallelSolver.h"
#include "../common/Result.h"
#include "../common/Definitions.h"

#include <fstream>

representations::ellpack::ELLPACK tools::invokers::ellpack::ELLPACKInvoker::loadELLPACK()
{
	std::fstream is;
	is.open(inputFile, std::fstream::in);
	representations::ellpack::ELLPACK ellpack;
	is >> ellpack;
	is.close();

	return ellpack;
}

FLOATING_TYPE * tools::invokers::ellpack::ELLPACKInvoker::createVectorB(int n)
{
	FLOATING_TYPE *b = new FLOATING_TYPE[n];
	for (int i = 0; i < n; i++)
		b[i] = 1;

	return b;
}

void tools::invokers::ellpack::ELLPACKInvoker::saveResult(representations::result::Result & result)
{
	std::string metadataFile = this->outputFile + META_EXTENSION;
	std::string outputFile = this->outputFile + OUTPUT_EXTENSION;
	result.save(metadataFile, outputFile);
}

tools::invokers::ellpack::ELLPACKInvoker::ELLPACKInvoker(std::string inputFile, std::string outputFile, int threads, int iterations)
	: inputFile(inputFile), outputFile(outputFile), threads(threads), iterations(iterations)
{
}

void tools::invokers::ellpack::ELLPACKInvoker::invoke()
{
	representations::ellpack::ELLPACK ellpack = loadELLPACK();
	FLOATING_TYPE *b = createVectorB(ellpack.N);

	representations::result::Result result;
	solvers::parallel::ellpack::ELLPACKParallelSolver solver;
	representations::output::Output output;
	auto timer = tools::measurements::timers::ExecutionTimer();
	int numberOfThreads = threads;

	std::function<void()> solveCSRparallelRoutine = [&output, &solver, &ellpack, &b, &numberOfThreads]()
	{
		output = solver.solve(ellpack, b, numberOfThreads);
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
