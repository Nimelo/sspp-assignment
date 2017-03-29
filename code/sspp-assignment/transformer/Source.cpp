#include "../common/CommandLineParameterReader.h"

#include <iostream>
#include <fstream>
#include <string>
#include "../common/MatrixMarketStream.h"
#include "../common/UnsignedFloatReader.h"
#include "../common/CRS.h"
#include "../common/ELLPACK.h"

#define ARG_IN_FILE "-i"
#define ARG_OUT_FILE "-o"
#define FLAG_CRS "-crs"
#define FLAG_ELLPACK "-ellpack"

int main(int argc, const char** argv) {
  //TODO: Add handling unexpected error.
  try {
    using namespace sspp::common;
    std::vector<Argument> arguments =
    {
      Argument(ARG_IN_FILE, ArgumentType::Single),
      Argument(ARG_OUT_FILE, ArgumentType::Single),
      Argument(FLAG_CRS, ArgumentType::Flag),
      Argument(FLAG_ELLPACK, ArgumentType::Flag)
    };

    CommandLineParameterReader reader(arguments);
    reader.Load(argc, argv);

    if(reader.HasArgument(ARG_IN_FILE) && reader.HasArgument(ARG_OUT_FILE)
       && (reader.HasArgument(FLAG_CRS) || reader.HasArgument(FLAG_ELLPACK))) {
      std::string inputFile = reader.GetParameter(ARG_IN_FILE);
      std::fstream fs, in_fs;
      std::string outputFile = reader.GetParameter(ARG_OUT_FILE);

      in_fs.open(inputFile, std::fstream::in);
      UnsignedFlaotReader tuple_reader;
      MatrixMarketStream<float> mms(in_fs, tuple_reader);

      if(reader.HasArgument(FLAG_CRS)) {
        CRS<float> crs(mms);
        fs.open(outputFile + ".crs", std::fstream::out | std::fstream::trunc);
        fs << crs;
        fs.close();
      }

      if(reader.HasArgument(FLAG_ELLPACK)) {
        ELLPACK<float> ellpack(mms);
        fs.open(outputFile + ".ellpack", std::fstream::out | std::fstream::trunc);
        fs << ellpack;
        fs.close();
      }

      in_fs.close();
    } else {
      std::cout << "Incorrect command line paramteres!\n";
    }
  } catch(...) {
    std::cout << "Unexpected error occured!\n";
  }

  return 0;
}