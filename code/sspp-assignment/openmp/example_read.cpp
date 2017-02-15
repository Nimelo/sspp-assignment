#include "../common/MatrixMarketReader.h"
#include "../common/IntermediarySparseMatrix.h"
#include "../common/CSRTransformer.h"
#include "../common/CSRSolver.h"
#include "../common/Definitions.h"
#include "../common/ExecutionTimer.h"
#include <vld.h>
#include <iostream>

int main(int argc, char *argv[])
{
	{	
		auto timer = tools::measurements::timers::ExecutionTimer();

		io::readers::MatrixMarketReader matrixReader;
		representations::intermediary::IntermediarySparseMatrix ism;

		std::function<void()> function = [&ism, &matrixReader, argv]()
		{
			ism = matrixReader.fromFile(argv[1]);
		};

		auto executionTime = timer.measure(function);

		tools::transformers::csr::CSRTransformer csrTransformer;
		representations::csr::CSR csr = 
		(csrTransformer.transform(ism));
		
		tools::solvers::csr::CSRSolver csrSolver;
		FLOATING_TYPE *x = new FLOATING_TYPE[ism.N];
		for (int i = 0; i < ism.N; i++)
			x[i] = 1;

		representations::output::Output output(csrSolver.solve(csr, x));

		delete[] x;
	}
}
