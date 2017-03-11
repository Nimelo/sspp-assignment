#ifndef __H_CSR_INVOKER
#define __H_CSR_INVOKER

#include <string>
#include "..\common\CSR.h"
#include "..\common\Definitions.h"
#include "..\common\Result.h"

namespace tools
{
	namespace invokers
	{
		namespace csr
		{
			class CSRInvoker
			{
			protected:
				std::string inputFile;
				std::string outputFile;
				int threads;
				int iterations;

				representations::csr::CSR loadCSR();
				FLOATING_TYPE *createVectorB(int n);
				void saveResult(representations::result::Result & result);
			public:
				CSRInvoker(std::string inputFile, std::string outputFile, int threads, int iterations);
				void invoke();
			};
		}
	}
}

#endif
