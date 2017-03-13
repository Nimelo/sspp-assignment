#include "ELLPACKInvoker.h"
#include "../common/ExecutionTimer.h"
#include "../openmp/ELLPACKOpenMPSolver.h"
#include "../common/Result.h"
#include "../common/SingleHeader.h"
#include "../common/Definitions.h"
#include "../common/ELLPACKSolver.h"

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
	std::string outputFile = this->outputFile + DASH_ELLPACK + OUTPUT_EXTENSION;
	std::fstream fs;
	fs.open(outputFile, std::fstream::out | std::fstream::trunc);
	fs << result;
	fs.close();
}

tools::invokers::ellpack::ELLPACKInvoker::ELLPACKInvoker(std::string inputFile, std::string outputFile, int threads, int iterationsParallel, int iterationsSerial)
	: inputFile(inputFile), outputFile(outputFile), threads(threads), iterationsParallel(iterationsParallel), iterationsSerial(iterationsSerial)
{
}

void tools::invokers::ellpack::ELLPACKInvoker::invoke()
{
	representations::ellpack::ELLPACK ellpack = loadELLPACK();
	FLOATING_TYPE *b = createVectorB(ellpack.N);

	representations::result::Result result;
	tools::solvers::ellpack::ELLPACKOpenMPSolver parallelsSolver;
	tools::solvers::ellpack::ELLPACKSolver solver;

	representations::output::Output output;
	auto timer = tools::measurements::timers::ExecutionTimer();
	int numberOfThreads = threads;

	std::function<void()> solveCSRSerialRoutine = [&output, &solver, &ellpack, &b]()
	{
		output = solver.solve(ellpack, b);
	};

	std::function<void()> solveCSRparallelRoutine = [&output, &parallelsSolver, &ellpack, &b, &numberOfThreads]()
	{
		parallelsSolver.setThreads(numberOfThreads);
		output = parallelsSolver.solve(ellpack, b);
	};

	for (int i = 0; i < iterationsSerial; i++)
	{
		auto executionTime = timer.measure(solveCSRSerialRoutine);
		result.serialResult.executionTimes.push_back(executionTime.count());
	}

	for (int i = 0; i < iterationsParallel; i++)
	{
		auto executionTime = timer.measure(solveCSRparallelRoutine);
		result.parallelResult.executionTimes.push_back(executionTime.count());
	}

	result.parallelResult.output = output;

	saveResult(result);

	delete[] b;
}
