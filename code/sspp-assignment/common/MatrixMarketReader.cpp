#include "MatrixMarketReader.h"
#include "ReadMatrixException.h"
#include "mmio.h"

representations::intermediary::IntermediarySparseMatrix io::readers::MatrixMarketReader::fromFile(const char * fileName)
{
	int ret_code;
	MM_typecode matcode;
	FILE *f;
	int M, N, nz;
	int i, *I, *J;
	double *val;

	if ((f = fopen(fileName, "r")) == NULL)
		throw io::exceptions::ReadMatrixException();
	
	if (mm_read_banner(f, &matcode) != 0)
	{
		printf("Could not process Matrix Market banner.\n");
		throw io::exceptions::ReadMatrixException();
	}


	/*  This is how one can screen matrix types if their application */
	/*  only supports a subset of the Matrix Market data types.      */

	if (mm_is_complex(matcode) && mm_is_matrix(matcode) &&
		mm_is_sparse(matcode))
	{
		printf("Sorry, this application does not support ");
		printf("Market Market type: [%s]\n", mm_typecode_to_str(matcode));
		throw io::exceptions::ReadMatrixException();
	}

	/* find out size of sparse matrix .... */

	if ((ret_code = mm_read_mtx_crd_size(f, &M, &N, &nz)) != 0)
		throw io::exceptions::ReadMatrixException();


	/* reseve memory for matrices */

	I = new int[nz];
	J = new int[nz];
	val = new double[nz];


	/* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
	/*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
	/*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */

	for (i = 0; i<nz; i++)
	{
		fscanf(f, "%d %d %lg\n", &I[i], &J[i], &val[i]);
		I[i]--;  /* adjust from 1-based to 0-based */
		J[i]--;
	}

	if (f != stdin) fclose(f);

	return representations::intermediary::IntermediarySparseMatrix(M, N, nz, I, J, val);
}
