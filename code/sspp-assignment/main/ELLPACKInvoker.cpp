#include "ELLPACKInvoker.h"
#include "../common/ExecutionTimer.h"
#include "../openmp/ELLPACKOpenMPSolver.h"
#include "../common/Result.h"
#include "../common/SingleHeader.h"
#include "../common/Definitions.h"
#include "../common/ELLPACKSolver.h"

#include <fstream>

sspp::representations::ELLPACK sspp::tools::invokers::ELLPACKInvoker::loadELLPACK() {
  std::fstream is;
  is.open(inputFile, std::fstream::in);
  representations::ELLPACK ellpack;
  is >> ellpack;
  is.close();

  return ellpack;
}

std::vector<FLOATING_TYPE> sspp::tools::invokers::ELLPACKInvoker::createVectorB(int n) {
  std::vector<FLOATING_TYPE> b(n);
  for(int i = 0; i < n; i++)
    b[i] = 1;

  return b;
}

void sspp::tools::invokers::ELLPACKInvoker::saveResult(representations::result::Result & result) {
  std::string outputFile = this->outputFile + DASH_ELLPACK + OUTPUT_EXTENSION;
  std::fstream fs;
  fs.open(outputFile, std::fstream::out | std::fstream::trunc);
  fs << result;
  fs.close();
}

sspp::tools::invokers::ELLPACKInvoker::ELLPACKInvoker(std::string inputFile, std::string outputFile, int iterationsParallel, int iterationsSerial)
  : inputFile(inputFile), outputFile(outputFile), iterationsParallel(iterationsParallel), iterationsSerial(iterationsSerial) {
}

void sspp::tools::invokers::ELLPACKInvoker::invoke(solvers::AbstractELLPACKSolver & parallelSolver) {
  representations::ELLPACK ellpack = loadELLPACK();
  std::vector<FLOATING_TYPE> b = createVectorB(ellpack.GetColumns());

  representations::result::Result result;
  tools::solvers::ELLPACKSolver solver;

  representations::Output output;
  auto timer = tools::measurements::ExecutionTimer();

  std::function<void()> solveCSRSerialRoutine = [&output, &solver, &ellpack, &b]() {
    output = solver.Solve(ellpack, b);
  };

  std::function<void()> solveCSRparallelRoutine = [&output, &parallelSolver, &ellpack, &b]() {
    output = parallelSolver.Solve(ellpack, b);
  };

  for(int i = 0; i < iterationsSerial; i++) {
    auto executionTime = timer.measure(solveCSRSerialRoutine);
    result.GetSerial().GetExecutionTimes().push_back(executionTime.count());
  }

  for(int i = 0; i < iterationsParallel; i++) {
    auto executionTime = timer.measure(solveCSRparallelRoutine);
    result.GetParallel().GetExecutionTimes().push_back(executionTime.count());
  }

  result.GetParallel().GetOutput() = output;

  saveResult(result);
}
