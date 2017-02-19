#ifndef __H_CSR
#define __H_CSR

#include "Definitions.h"

namespace representations
{
	namespace csr
	{
		class CSR
		{
			public:
				int NZ;
				int M;
				int N;
				int *IRP;
				int *JA;
				FLOATING_TYPE *AS;
			protected:
				void rewrite(CSR & lhs, const CSR & rhs);
			public:
				CSR();
				CSR(int NZ, int M, int N, int *IRP, int *JA, FLOATING_TYPE *AS);
				CSR(const CSR &other);
				CSR & operator=(CSR rhs);
				~CSR();
				int getIRPSize() const;
				int getJASize() const;
				int getASSize() const;
		};
	}
}

#endif