#include "../common/CommandLineParameterReader.h"
#include "../common/MatrixMarketReader.h"
#include "../common/CSRTransformer.h"
#include "../common/ELLPACKTransformer.h"
#include "../common/Definitions.h"

#include <iostream>
#include <fstream>
#include <string>

#define TAG_IN_FILE "-i"
#define TAG_OUT_FILE "-o"
#define ARGUMENT_CSR "-csr"
#define ARGUMENT_ELLPACK "-ellpack"

int main(int argc, const char** argv)
{
	//TODO: Add handling unexpected error.
	using namespace io::readers::input::commandline;
	std::vector<arguments::Argument> arguments = 
	{ 
		arguments::Argument(TAG_IN_FILE, arguments::ArgumentType::Single),
		arguments::Argument(TAG_OUT_FILE, arguments::ArgumentType::Single),
		arguments::Argument(ARGUMENT_CSR, arguments::ArgumentType::Flag),
		arguments::Argument(ARGUMENT_ELLPACK, arguments::ArgumentType::Flag)
	};

	CommandLineParameterReader reader(arguments);
	reader.load(argc, argv);

	if (reader.hasArgument(TAG_IN_FILE) && reader.hasArgument(TAG_OUT_FILE)
		&& (reader.hasArgument(ARGUMENT_CSR) || reader.hasArgument(ARGUMENT_ELLPACK)))
	{

		io::readers::MatrixMarketReader mmr;
		std::string inputFile = reader.get(TAG_IN_FILE);
		auto ism = mmr.fromFile(inputFile.c_str());
		std::fstream fs;
		std::string outputFile = reader.get(TAG_OUT_FILE);

		if (reader.hasArgument(ARGUMENT_CSR))
		{
			tools::transformers::csr::CSRTransformer csrTransformer;
			auto csr = csrTransformer.transform(ism);
			fs.open(outputFile + CSR_EXTENSION, std::fstream::out | std::fstream::trunc);
			fs << csr;
			fs.close();
		}

		if (reader.hasArgument(ARGUMENT_ELLPACK))
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