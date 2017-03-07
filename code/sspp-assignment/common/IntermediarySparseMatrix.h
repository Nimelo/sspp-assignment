#ifndef __H_INETERMEDIARY_SPARSE_MATRIX
#define __H_INETERMEDIARY_SPARSE_MATRIX

#include "Definitions.h"

namespace representations
{
	namespace intermediary
	{
		class IntermediarySparseMatrix
		{
		public:
			int NZ;
			int M;
			int N;
			int *IIndexes;
			int *JIndexes;
			FLOATING_TYPE *Values;
		public:
			IntermediarySparseMatrix();
			IntermediarySparseMatrix(int m, int n, int nz, int *iIndexes, int *jIndexes, FLOATING_TYPE *values);
			IntermediarySparseMatrix(const IntermediarySparseMatrix &other);
			IntermediarySparseMatrix & operator=(IntermediarySparseMatrix rhs);
			~IntermediarySparseMatrix();
		};
	}
}

#endif