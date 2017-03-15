#include "../common/CommandLineParameterReader.h"
#include "CSRInvoker.h"
#include "ELLPACKInvoker.h"
#include "../common/Definitions.h"
#include "../openmp/CSROpenMPSolver.h"
#include "../openmp/ELLPACKOpenMPSolver.h"
#include "../cuda/CSRCudaSolver.h"
#include "../cuda/ELLPACKCudaSolver.h"

#include <iostream>
#include <fstream>
#include <string>

#define ARG_IN_FILE "-i"
#define ARG_OUT_FILE "-o"
#define ARG_THREADS "-threads"
#define ARG_ITERATIONS_PARALLEL "-ip"
#define ARG_ITERATIONS_SERIAL "-is"
#define FLAG_CSR "-csr"
#define FLAG_ELLPACK "-ellpack"
#define FLAG_CUDA "-cuda"
#define FLAG_OPENMP "-openmp"

int main(int argc, const char** argv) {
  //TODO: Add handling unexpected error.
  using namespace sspp::io::readers::commandline;
  std::vector<Argument> arguments =
  {
    Argument(ARG_IN_FILE, ArgumentType::Single),
    Argument(ARG_OUT_FILE, ArgumentType::Single),
    Argument(ARG_THREADS, ArgumentType::Single),
    Argument(ARG_ITERATIONS_PARALLEL, ArgumentType::Single),
    Argument(ARG_ITERATIONS_SERIAL, ArgumentType::Single),
    Argument(FLAG_CSR, ArgumentType::Flag),
    Argument(FLAG_ELLPACK, ArgumentType::Flag),
    Argument(FLAG_CUDA, ArgumentType::Flag),
    Argument(FLAG_OPENMP, ArgumentType::Flag)
  };

  CommandLineParameterReader reader(arguments);
  reader.load(argc, argv);

  if(reader.hasArgument(ARG_IN_FILE)
     && reader.hasArgument(ARG_OUT_FILE)
     && reader.hasArgument(ARG_THREADS)
     && reader.hasArgument(ARG_ITERATIONS_PARALLEL)
     && reader.hasArgument(ARG_ITERATIONS_SERIAL)
     && ((reader.hasArgument(FLAG_CSR) && !reader.hasArgument(FLAG_ELLPACK))
         || !reader.hasArgument(FLAG_CSR) && reader.hasArgument(FLAG_ELLPACK))
     && ((reader.hasArgument(FLAG_OPENMP) && !reader.hasArgument(FLAG_CUDA))
         || !reader.hasArgument(FLAG_OPENMP) && reader.hasArgument(FLAG_CUDA))) {
    std::string inputFile = reader.get(ARG_IN_FILE);
    std::string outputFile = reader.get(ARG_OUT_FILE);
    int threads = reader.get(ARG_THREADS);
    int iterationsParallel = reader.get(ARG_ITERATIONS_PARALLEL);
    int iterationsSerial = reader.get(ARG_ITERATIONS_SERIAL);

    using namespace sspp::tools::solvers;
    //TODO: Add support for cuda code.
    if(reader.hasArgument(FLAG_CSR)) {
      sspp::tools::invokers::CSRInvoker csrInvoker(inputFile, outputFile, iterationsParallel, iterationsSerial);
      if(reader.hasArgument(FLAG_CUDA)) {
        csrInvoker.invoke(CSRCudaSolver());
      } else {
        csrInvoker.invoke(CSROpenMPSolver(threads));
      }
    } else {
      sspp::tools::invokers::ELLPACKInvoker ellpackInvoker(inputFile, outputFile, iterationsParallel, iterationsSerial);
      if(reader.hasArgument(FLAG_CUDA)) {
        ellpackInvoker.invoke(ELLPACKCudaSolver());
      } else {
        ellpackInvoker.invoke(ELLPACKOpenMPSolver(threads));
      }
    }
  } else {
    std::cout << "Incorrect command line paramteres!";
  }

  return 0;
}