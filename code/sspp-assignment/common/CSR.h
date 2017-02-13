#ifndef __H_CSR
#define __H_CSR

#include "Definitions.h"

namespace representations
{
	namespace csr
	{
		class CSR
		{
			protected:
				int NZ;
				int M;
				int N;
				int *IRP;
				int *JA;
				FLOATING_TYPE *AS;
			public:
				CSR(int NZ, int M, int N, int *IRP, int *JA, FLOATING_TYPE *AS);
				~CSR();
				int getM() const;
				int getN() const;
				int * getIRP() const;
				int * getJA() const;
				FLOATING_TYPE * getAS() const;
				int getIRPSize() const;
				int getJASize() const;
				int getASSize() const;
		};
	}
}

#endif