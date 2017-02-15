#ifndef __H_CSR_TRANSFORMER
#define __H_CSR_TRANSFORMER

#include "CSR.h"
#include "IntermediarySparseMatrix.h"

namespace tools
{
	namespace transformers
	{
		namespace csr
		{
			class CSRTransformer
			{
				public:
					representations::csr::CSR transform(representations::intermediary::IntermediarySparseMatrix & ism);
			};
		}
	}
}

#endif