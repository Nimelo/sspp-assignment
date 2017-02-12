#include "../common/MatrixMarketReader.h"
#include "../common/IntermediarySparseMatrix.h"
#include <vld.h>
#include <iostream>

int main(int argc, char *argv[])
{
	io::readers::MatrixMarketReader matrixReader;
	representations::intermediary::IntermediarySparseMatrix matrix = matrixReader.fromFile(argv[1]);

	for (size_t i = 0; i < matrix.getNZ(); i++)
	{
		std::cout << matrix.getIIndexes()[i] << " " << matrix.getJIndexes()[i] << " " << matrix.getValues()[i] << std::endl;
	}
}
