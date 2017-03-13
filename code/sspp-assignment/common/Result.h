#ifndef __H_RESULT
#define __H_RESULT

#include "Definitions.h"
#include "Output.h"
#include "SingleHeader.h"

#include <ostream>
#include <string>
#include <vector>

namespace representations
{
	namespace result
	{
		class Result
		{
		public:
			representations::result::single::SingleResult serialResult;
			representations::result::single::SingleResult parallelResult;
			friend std::ostream& operator <<(std::ostream& os, const Result & result);
		};
	}
}

#endif
