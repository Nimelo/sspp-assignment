#ifndef __H_ELLPACK_TRANSFORMER
#define __H_ELLPACK_TRANSFORMER

#include "IntermediarySparseMatrix.h"
#include "ELLPACK.h"
#include "Definitions.h"

namespace tools
{
	namespace transformers
	{
		namespace ellpack
		{
			class ELLPACKTransformer
			{
				protected:
					void preprocessISM(const representations::intermediary::IntermediarySparseMatrix & ism);
					int * findAuxilliaryArray(const representations::intermediary::IntermediarySparseMatrix & ism);
					void allocateArrays(int ***JA, FLOATING_TYPE ***AS, int M, int MAXNZ);
					representations::ellpack::ELLPACK transformImpl(const representations::intermediary::IntermediarySparseMatrix & ism, int M, int MAXNZ, int **JA, FLOATING_TYPE **AS, int *auxArray);
				public:
					representations::ellpack::ELLPACK transform(const representations::intermediary::IntermediarySparseMatrix & ism);
			};
		}
	}
}

#endif