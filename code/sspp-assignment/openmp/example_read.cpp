#include "../common/MatrixMarketReader.h"
#include "../common/IntermediarySparseMatrix.h"
#include "../common/CSRTransformer.h"
#include "../common/CSRSolver.h"
#include "../common/Definitions.h"
#include <vld.h>
#include <iostream>

int main(int argc, char *argv[])
{
	io::readers::MatrixMarketReader matrixReader;
	representations::intermediary::IntermediarySparseMatrix ism = matrixReader.fromFile(argv[1]);

	/*for (size_t i = 0; i < ism.getNZ(); i++)
	{
		std::cout << ism.getIIndexes()[i] << " " << ism.getJIndexes()[i] << " " << ism.getValues()[i] << std::endl;
	}*/

	tools::transformers::csr::CSRTransformer csrTransformer;
	representations::csr::CSR csr(csrTransformer.transform(ism));

	tools::solvers::csr::CSRSolver csrSolver;
	FLOATING_TYPE *x = new FLOATING_TYPE[ism.getN()];

	representations::output::Output output(csrSolver.solve(csr, x));

	delete [] x;
}
