#include "../common/CommandLineParameterReader.h"

#include <iostream>
#include <fstream>
#include <string>
#include "../common/CRS.h"
#include "../common/ELLPACK.h"
#include "../common/MetaPerformanceResult.h"
#include "CRSRunner.h"
#include "ELLPACKRunner.h"

#define ARG_IN_FILE "-i"
#define ARG_OUT_FILE "-o"
#define FLAG_CRS "-crs"
#define FLAG_ELLPACK "-ellpack"
#define ARG_ITERATIONS "-iter"

int main(int argc, const char** argv) {
  //TODO: Add handling unexpected error.
  try {
    using namespace sspp::common;
    std::vector<Argument> arguments =
    {
      Argument(ARG_IN_FILE, ArgumentType::Single),
      Argument(ARG_OUT_FILE, ArgumentType::Single),
      Argument(FLAG_CRS, ArgumentType::Flag),
      Argument(FLAG_ELLPACK, ArgumentType::Flag),
      Argument(ARG_ITERATIONS, Single)
    };

    CommandLineParameterReader reader(arguments);
    reader.Load(argc, argv);

    if(reader.HasArgument(ARG_IN_FILE) && reader.HasArgument(ARG_OUT_FILE)
       && (reader.HasArgument(FLAG_CRS) || reader.HasArgument(FLAG_ELLPACK))
       && reader.HasArgument(ARG_ITERATIONS)) {
      unsigned iterations = reader.GetParameter(ARG_ITERATIONS);
      std::string inputFile = reader.GetParameter(ARG_IN_FILE);
      std::fstream fs;
      std::string outputFile = reader.GetParameter(ARG_OUT_FILE);
      fs.open(inputFile, std::fstream::in);
      if(reader.HasArgument(FLAG_CRS)) {
        CRS<float> crs;
        fs >> crs;
        fs.close();
        fs.open(outputFile + ".meta-crs", std::fstream::out | std::fstream::trunc);
        fs << sspp::serial::CRSRunner::run<float>(crs, iterations);
        fs.close();
      } else if(reader.HasArgument(FLAG_ELLPACK)) {
        ELLPACK<float> ellpack;
        fs >> ellpack;
        fs.close();
        fs.open(outputFile + ".meta-ellpack", std::fstream::out | std::fstream::trunc);
        fs << sspp::serial::ELLPACKRunner::run<float>(ellpack, iterations);
        fs.close();
      }
    } else {
      std::cout << "Incorrect command line paramteres!\n";
    }
  } catch(...) {
    std::cout << "Unexpected error occured!\n";
  }

  return 0;
}
