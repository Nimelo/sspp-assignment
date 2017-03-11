#include "../common/CommandLineParameterReader.h"
#include "CSRInvoker.h"
#include "ELLPACKInvoker.h"
#include "../common/Definitions.h"

#include <iostream>
#include <fstream>
#include <string>

#define TAG_IN_FILE "-i"
#define TAG_OUT_FILE "-o"
#define TAG_THREADS "-threads"
#define TAG_ITERATIONS "-iterations"
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
		arguments::Argument(TAG_THREADS, arguments::ArgumentType::Single),
		arguments::Argument(TAG_ITERATIONS, arguments::ArgumentType::Single),
		arguments::Argument(ARGUMENT_CSR, arguments::ArgumentType::Flag),
		arguments::Argument(ARGUMENT_ELLPACK, arguments::ArgumentType::Flag)
	};

	CommandLineParameterReader reader(arguments);
	reader.load(argc, argv);

	if (reader.hasArgument(TAG_IN_FILE)
		&& reader.hasArgument(TAG_OUT_FILE)
		&& reader.hasArgument(TAG_THREADS)
		&& reader.hasArgument(TAG_ITERATIONS)
		&& ((reader.hasArgument(ARGUMENT_CSR) && !reader.hasArgument(ARGUMENT_ELLPACK))
			|| !reader.hasArgument(ARGUMENT_CSR) && reader.hasArgument(ARGUMENT_ELLPACK)))
	{
		std::string inputFile = reader.get(TAG_IN_FILE);
		std::string outputFile = reader.get(TAG_OUT_FILE);
		int threads = reader.get(TAG_THREADS);
		int iterations = reader.get(TAG_ITERATIONS);

		if (reader.hasArgument(ARGUMENT_CSR))
		{
			tools::invokers::csr::CSRInvoker csrInvoker(inputFile, outputFile, threads, iterations);
			csrInvoker.invoke();
		}
		else
		{
			tools::invokers::ellpack::ELLPACKInvoker ellpackInvoker(inputFile, outputFile, threads, iterations);
			ellpackInvoker.invoke();
		}
	}
	else
	{
		std::cout << "Incorrect command line paramteres!";
	}

	return 0;
}