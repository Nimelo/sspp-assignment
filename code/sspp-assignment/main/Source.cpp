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

int main(int argc, const char** argv)
{
	//TODO: Add handling unexpected error.
	using namespace io::readers::input::commandline;
	std::vector<arguments::Argument> arguments =
	{
		arguments::Argument(ARG_IN_FILE, arguments::ArgumentType::Single),
		arguments::Argument(ARG_OUT_FILE, arguments::ArgumentType::Single),
		arguments::Argument(ARG_THREADS, arguments::ArgumentType::Single),
		arguments::Argument(ARG_ITERATIONS_PARALLEL, arguments::ArgumentType::Single),
		arguments::Argument(ARG_ITERATIONS_SERIAL, arguments::ArgumentType::Single),
		arguments::Argument(FLAG_CSR, arguments::ArgumentType::Flag),
		arguments::Argument(FLAG_ELLPACK, arguments::ArgumentType::Flag),
		arguments::Argument(FLAG_CUDA, arguments::ArgumentType::Flag),
		arguments::Argument(FLAG_OPENMP, arguments::ArgumentType::Flag)
	};

	CommandLineParameterReader reader(arguments);
	reader.load(argc, argv);

	if (reader.hasArgument(ARG_IN_FILE)
		&& reader.hasArgument(ARG_OUT_FILE)
		&& reader.hasArgument(ARG_THREADS)
		&& reader.hasArgument(ARG_ITERATIONS_PARALLEL)
		&& reader.hasArgument(ARG_ITERATIONS_SERIAL)
		&& ((reader.hasArgument(FLAG_CSR) && !reader.hasArgument(FLAG_ELLPACK))
			|| !reader.hasArgument(FLAG_CSR) && reader.hasArgument(FLAG_ELLPACK))
		&& ((reader.hasArgument(FLAG_OPENMP) && !reader.hasArgument(FLAG_CUDA))
			|| !reader.hasArgument(FLAG_OPENMP) && reader.hasArgument(FLAG_CUDA)))
	{
		std::string inputFile = reader.get(ARG_IN_FILE);
		std::string outputFile = reader.get(ARG_OUT_FILE);
		int threads = reader.get(ARG_THREADS);
		int iterationsParallel = reader.get(ARG_ITERATIONS_PARALLEL);
		int iterationsSerial = reader.get(ARG_ITERATIONS_SERIAL);
		
		using namespace tools::solvers;
		//TODO: Add support for cuda code.
		if (reader.hasArgument(FLAG_CSR))
		{			
			tools::invokers::csr::CSRInvoker csrInvoker(inputFile, outputFile, iterationsParallel, iterationsSerial);
			if(reader.hasArgument(FLAG_CUDA))
			{
				csrInvoker.invoke(csr::CSRCudaSolver());
			}
			else
			{
				csrInvoker.invoke(csr::CSROpenMPSolver(threads));
			}
		}
		else
		{
			tools::invokers::ellpack::ELLPACKInvoker ellpackInvoker(inputFile, outputFile, iterationsParallel, iterationsSerial);
			if (reader.hasArgument(FLAG_CUDA))
			{
				ellpackInvoker.invoke(ellpack::ELLPACKCudaSolver());
			}
			else
			{
				ellpackInvoker.invoke(ellpack::ELLPACKOpenMPSolver(threads));
			}
		}
	}
	else
	{
		std::cout << "Incorrect command line paramteres!";
	}

	return 0;
}