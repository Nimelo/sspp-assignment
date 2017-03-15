#include "../common/CommandLineParameterReader.h"
#include "../common/MatrixMarketReader.h"
#include "../common/CSRTransformer.h"
#include "../common/ELLPACKTransformer.h"
#include "../common/Definitions.h"

#include <iostream>
#include <fstream>
#include <string>

#define ARG_IN_FILE "-i"
#define ARG_OUT_FILE "-o"
#define FLAG_CSR "-csr"
#define FLAG_ELLPACK "-ellpack"

int main(int argc, const char** argv) {
  //TODO: Add handling unexpected error.
  using namespace sspp::io::readers::commandline;
  std::vector<Argument> arguments =
  {
    Argument(ARG_IN_FILE, ArgumentType::Single),
    Argument(ARG_OUT_FILE, ArgumentType::Single),
    Argument(FLAG_CSR, ArgumentType::Flag),
    Argument(FLAG_ELLPACK, ArgumentType::Flag)
  };

  sspp::io::readers::commandline::CommandLineParameterReader reader(arguments);
  reader.load(argc, argv);

  if(reader.hasArgument(ARG_IN_FILE) && reader.hasArgument(ARG_OUT_FILE)
     && (reader.hasArgument(FLAG_CSR) || reader.hasArgument(FLAG_ELLPACK))) {
    sspp::io::readers::MatrixMarketReader mmr;
    std::string inputFile = reader.get(ARG_IN_FILE);
    auto ism = mmr.fromFile(inputFile.c_str());
    std::fstream fs;
    std::string outputFile = reader.get(ARG_OUT_FILE);

    if(reader.hasArgument(FLAG_CSR)) {
      sspp::tools::transformers::CSRTransformer csrTransformer;
      auto csr = csrTransformer.transform(ism);
      fs.open(outputFile + CSR_EXTENSION, std::fstream::out | std::fstream::trunc);
      fs << csr;
      fs.close();
    }

    if(reader.hasArgument(FLAG_ELLPACK)) {
      sspp::tools::transformers::ELLPACKTransformer ellpackTransformer;
      auto ellpack = ellpackTransformer.transform(ism);
      fs.open(outputFile + ELLPACK_EXTENSION, std::fstream::out | std::fstream::trunc);
      fs << ellpack;
      fs.close();
    }
  } else {
    std::cout << "Incorrect command line paramteres!";
  }

  return 0;
}