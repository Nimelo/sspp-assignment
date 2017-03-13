#include "MatrixMarketReader.h"
#include "InPlaceStableSorter.h"
#include "ReadMatrixException.h"
#include "mmio.h"

representations::intermediary::IntermediarySparseMatrix io::readers::MatrixMarketReader::fromFile(const char * fileName)
{
	int M, N, nz, *I, *J;
	FLOATING_TYPE *val;

	int result = mm_read_sparse(fileName, &M, &N, &nz, &val, &I, &J);

	return representations::intermediary::IntermediarySparseMatrix(M, N, nz, I, J, val);
}
