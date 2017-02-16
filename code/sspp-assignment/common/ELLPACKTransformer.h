#ifndef __H_ELLPACK_TRANSFORMER
#define __H_ELLPACK_TRANSFORMER

#include "IntermediarySparseMatrix.h"
#include "ELLPACK.h"

namespace tools
{
	namespace transformers
	{
		namespace ellpack
		{
			class ELLPACKTransformer
			{
				public:
					representations::ellpack::ELLPACK transform(const representations::intermediary::IntermediarySparseMatrix & ism);
			};
		}
	}
}

#endif