#ifndef __H_MATRIX_MARKET_READER
#define __H_MATRIX_MARKET_READER

#include "IntermediarySparseMatrix.h"

namespace io
{
	namespace readers
	{
		class MatrixMarketReader
		{
			public:
				representations::intermediary::IntermediarySparseMatrix fromFile(const char * fileName);
		};
	}
}

#endif