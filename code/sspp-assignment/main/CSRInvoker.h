#ifndef __H_CSR_INVOKER
#define __H_CSR_INVOKER

#include <string>
#include "..\common\CSR.h"
#include "..\common\Definitions.h"
#include "..\common\Result.h"
#include "..\common/AbstractCSRSolver.h"

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
				int iterationsParallel;
				int iterationsSerial;

				representations::csr::CSR loadCSR();
				FLOATING_TYPE *createVectorB(int n);
				void saveResult(representations::result::Result & result);
			public:
				CSRInvoker(std::string inputFile, std::string outputFile, int iterationsParallel, int iterationsSerial);
				void invoke(solvers::csr::AbstractCSRSolver & solver);
			};
		}
	}
}

#endif
