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

int main(int argc, const char** argv)
{
	//TODO: Add handling unexpected error.
	using namespace io::readers::input::commandline;
	std::vector<arguments::Argument> arguments = 
	{ 
		arguments::Argument(ARG_IN_FILE, arguments::ArgumentType::Single),
		arguments::Argument(ARG_OUT_FILE, arguments::ArgumentType::Single),
		arguments::Argument(FLAG_CSR, arguments::ArgumentType::Flag),
		arguments::Argument(FLAG_ELLPACK, arguments::ArgumentType::Flag)
	};

	CommandLineParameterReader reader(arguments);
	reader.load(argc, argv);

	if (reader.hasArgument(ARG_IN_FILE) && reader.hasArgument(ARG_OUT_FILE)
		&& (reader.hasArgument(FLAG_CSR) || reader.hasArgument(FLAG_ELLPACK)))
	{

		io::readers::MatrixMarketReader mmr;
		std::string inputFile = reader.get(ARG_IN_FILE);
		auto ism = mmr.fromFile(inputFile.c_str());
		std::fstream fs;
		std::string outputFile = reader.get(ARG_OUT_FILE);

		if (reader.hasArgument(FLAG_CSR))
		{
			tools::transformers::csr::CSRTransformer csrTransformer;
			auto csr = csrTransformer.transform(ism);
			fs.open(outputFile + CSR_EXTENSION, std::fstream::out | std::fstream::trunc);
			fs << csr;
			fs.close();
		}

		if (reader.hasArgument(FLAG_ELLPACK))
		{
			tools::transformers::ellpack::ELLPACKTransformer ellpackTransformer;
			auto ellpack = ellpackTransformer.transform(ism);
			fs.open(outputFile + ELLPACK_EXTENSION, std::fstream::out | std::fstream::trunc);
			fs << ellpack;
			fs.close();
		}
	}
	else
	{
		std::cout << "Incorrect command line paramteres!";
	}

	return 0;
}