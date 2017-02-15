#ifndef __H_ELLPACK
#define __H_ELLPACK

#include "Definitions.h"

namespace representations
{
	namespace ellpack
	{
		class ELLPACK
		{
			public:
				int M;
				int N;
				int NZ;
				int MAXNZ;
				int **JA;
				FLOATING_TYPE **AS;
		};
	}
}

#endif