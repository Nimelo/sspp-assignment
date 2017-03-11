#ifndef __H_SINGLE_RESULT
#define __H_SINGLE_RESULT

#include "Definitions.h"
#include "Output.h"

#include <ostream>
#include <string>
#include <vector>

namespace representations
{
	namespace result
	{
		namespace single
		{
			class SingleResult
			{
			public:
				representations::output::Output output;
				std::vector<double> executionTimes;
				friend std::ostream& operator <<(std::ostream& os, const SingleResult& result);
			};
		}
	}
}

#endif#pragma once
