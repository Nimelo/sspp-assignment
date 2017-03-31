//#include "CSRInvoker.h"
//#include "../common/ExecutionTimer.h"
//#include "../openmp/CSROpenMPSolver.h"
//#include "../common/CSRSolver.h"
//#include "../common/Result.h"
//#include "../common/Definitions.h"
//
//#include <fstream>
//
//sspp::representations::CSR sspp::tools::invokers::CSRInvoker::loadCSR() {
//  std::fstream is;
//  is.open(inputFile, std::fstream::in);
//  sspp::representations::CSR csr;
//  is >> csr;
//  is.close();
//
//  return csr;
//}
//
//std::vector<FLOATING_TYPE> sspp::tools::invokers::CSRInvoker::createVectorB(int n) {
//  std::vector<FLOATING_TYPE> b(n);
//  for(int i = 0; i < n; i++)
//    b[i] = 1;
//
//  return b;
//}
//
//void sspp::tools::invokers::CSRInvoker::saveResult(representations::result::Result & result) {
//  std::string outputFile = this->outputFile + DASH_CSR + OUTPUT_EXTENSION;
//  std::fstream fs;
//  fs.open(outputFile, std::fstream::out | std::fstream::trunc);
//  fs << result;
//  fs.close();
//}
//
//sspp::tools::invokers::CSRInvoker::CSRInvoker(std::string inputFile, std::string outputFile, int iterationsParallel, int iterationsSerial)
//  : inputFile(inputFile), outputFile(outputFile), iterationsParallel(iterationsParallel), iterationsSerial(iterationsSerial) {
//}
//
//void sspp::tools::invokers::CSRInvoker::invoke(sspp::tools::solvers::AbstractCSRSolver & parallelSolver) {
//  representations::CSR csr = loadCSR();
//  std::vector<FLOATING_TYPE> b = createVectorB(csr.GetColumns());
//
//  representations::result::Result result;
//  tools::solvers::CSRSolver serialSolver;
//
//  representations::Output output;
//  auto timer = tools::measurements::ExecutionTimer();
//
//  std::function<void()> solveCSRSerialRoutine = [&output, &serialSolver, &csr, &b]() {
//    output = serialSolver.Solve(csr, b);
//  };
//
//  std::function<void()> solveCSRparallelRoutine = [&output, &parallelSolver, &csr, &b]() {
//    output = parallelSolver.Solve(csr, b);
//  };
//
//  for(int i = 0; i < iterationsSerial; i++) {
//    auto executionTime = timer.measure(solveCSRSerialRoutine);
//    result.GetSerial().GetExecutionTimes().push_back(executionTime.count());
//  }
//  result.GetSerial().GetOutput() = output;
//
//  for(int i = 0; i < iterationsParallel; i++) {
//    auto executionTime = timer.measure(solveCSRparallelRoutine);
//    result.GetParallel().GetExecutionTimes().push_back(executionTime.count());
//  }
//  result.GetParallel().GetOutput() = output;
//
//  saveResult(result);
//}