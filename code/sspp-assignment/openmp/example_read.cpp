#include "../common/MatrixMarketReader.h"
#include "../common/IntermediarySparseMatrix.h"
#include "../common/CSRTransformer.h"
#include "../common/CSRSolver.h"
#include "../common/Definitions.h"
#include "../common/ExecutionTimer.h"
#include "../common/ELLPACKTransformer.h"
#include "CSRParallelSolver.h"
#ifdef WIN32
	#include <vld.h>
#endif
#include <iostream>
#include <omp.h>

#define OUTPUT_DIR_TAG "-o"
#define INPUT_DIR_TAG "-i"
#define FILES_TAG "-fs"
#define ITERATIONS_TAG "-iter"
#define M_TAG "-m"
#define N_TAG "-n"
#define TRANSFORM_TAG "-t"

#define TRANSFORM_ACTION "exe -transform -i "
int main(int argc, char *argv[])
{
	{	


		/**/
		auto timer = tools::measurements::timers::ExecutionTimer();

		// 1. Read MatrixMarketFromTheFile
		io::readers::MatrixMarketReader matrixReader;
		representations::intermediary::IntermediarySparseMatrix ism;
		std::function<void()> readingMatrixRoutine = [&ism, &matrixReader, argv]()
		{
			ism = matrixReader.fromFile(argv[1]);
		};

		// 2. Transform ISM to CSR
		tools::transformers::csr::CSRTransformer csrTransformer;
		representations::csr::CSR csr;
		std::function<void()> transformISMToCSRRoutine = [&csr, &csrTransformer, &ism]()
		{
			csr = csrTransformer.transform(ism);
		};

		// 3. Transform ISM to ELLPACK
		tools::transformers::ellpack::ELLPACKTransformer ellpackTransformer;
		representations::ellpack::ELLPACK ellpack;
		std::function<void()> transformISMToELLPACKRoutine = [&ellpack, &ellpackTransformer, &ism]()
		{
			ellpack = ellpackTransformer.transform(ism);
		};

		// 4. Create vector b
		FLOATING_TYPE *x;

		// 5. Solve CSR
		solvers::parallel::csr::CSRParallelSolver csrParallelSolver;
		representations::output::Output outputCSRParallel;
		std::function<void()> solveCSRparallelRoutine = [&outputCSRParallel, &csrParallelSolver, &csr, &x]()
		{
			outputCSRParallel = csrParallelSolver.solve(csr, x);
		};

		// 6. Solve ELLPACK

		// 5. Get serial execution time (CSR)
		tools::solvers::csr::CSRSolver csrSolver;
		representations::output::Output outputCSR;
		std::function<void()> solveCSRRoutine = [&outputCSR, &csrSolver, &csr, &x]()
		{
			outputCSR = csrSolver.solve(csr, x);
		};

		// *. Run routines and measure times
		auto readingMatrixRoutineTime = timer.measure(readingMatrixRoutine);
		x = new FLOATING_TYPE[ism.N];
		for (int i = 0; i < ism.N; i++)
			x[i] = 1;
		auto transformISMToCSRRoutineTime = timer.measure(transformISMToCSRRoutine);
		auto transformISMToEllpackRoutineTime = timer.measure(transformISMToELLPACKRoutine);
		auto solveCSRparallelRoutineTime = timer.measure(solveCSRparallelRoutine);
		auto solveCSRRoutineTime = timer.measure(solveCSRRoutine);

		std::function<int(int)> add1 = [](int x)
		{
			return x + 1;
		};
		delete[] x;
	}

	#pragma omp parallel
	{
		std::cout << std::endl << "Hello World!";
	}
	return 0;
}
