#ifndef __H_INETERMEDIARY_SPARSE_MATRIX
#define __H_INETERMEDIARY_SPARSE_MATRIX

#include "Definitions.h"

namespace representations 
{
	namespace intermediary
	{
		class IntermediarySparseMatrix
		{
			protected:
				int nz;
				int m;
				int n;
				int *i;
				int *j;
				FLOATING_TYPE *values;
			public:
				IntermediarySparseMatrix(int m, int n, int nz, int *iIndexes, int *jIndexes, FLOATING_TYPE *values);
				IntermediarySparseMatrix(const IntermediarySparseMatrix &other);
				~IntermediarySparseMatrix();
				int getNZ() const;
				int getM() const;
				int getN() const;
				int * getIIndexes() const;
				int * getJIndexes() const;
				FLOATING_TYPE * getValues() const;
		};
	}
}

#endif