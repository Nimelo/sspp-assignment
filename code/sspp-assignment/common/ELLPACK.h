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
			protected:
				static void rewrite(ELLPACK & lhs, const ELLPACK & rhs);
			public:
				ELLPACK();
				ELLPACK(int M, int N, int NZ, int MAXNZ, int **JA, FLOATING_TYPE **AS);
				ELLPACK(const ELLPACK & other);
				ELLPACK & operator=(ELLPACK rhs);
				~ELLPACK();
		};
	}
}

#endif