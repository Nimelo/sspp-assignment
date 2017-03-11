#ifndef __H_ELLPACK_INVOKER
#define __H_ELLPACK_INVOKER

#include <string>
#include "..\common\ELLPACK.h"
#include "..\common\Definitions.h"
#include "..\common\Result.h"

namespace tools
{
	namespace invokers
	{
		namespace ellpack
		{
			class ELLPACKInvoker
			{
			protected:
				std::string inputFile;
				std::string outputFile;
				int threads;
				int iterationsParallel;
				int iterationsSerial;

				representations::ellpack::ELLPACK loadELLPACK();
				FLOATING_TYPE *createVectorB(int n);
				void saveResult(representations::result::Result & result);
			public:
				ELLPACKInvoker(std::string inputFile, std::string outputFile, int threads, int iterationsParallel, int iterationsSerial);
				void invoke();
			};
		}
	}
}

#endif
