#ifndef __H_INETERMEDIARY_SPARSE_MATRIX
#define __H_INETERMEDIARY_SPARSE_MATRIX

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
				double *values;
			public:
				IntermediarySparseMatrix(int m, int n, int nz, int *iIndexes, int *jIndexes, double *values);
				IntermediarySparseMatrix(const IntermediarySparseMatrix &other);
				~IntermediarySparseMatrix();
				int getNZ() const;
				int getM() const;
				int getN() const;
				int * getIIndexes() const;
				int * getJIndexes() const;
				double * getValues() const;
		};
	}
}

#endif