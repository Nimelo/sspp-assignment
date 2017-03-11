#ifndef __H_RESULT
#define __H_RESULT

#include "Definitions.h"
#include "Output.h"

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
			representations::output::Output output;
			std::vector<double> executionTimes;
			void save(std::string metadata, std::string output);

			friend std::ostream& operator <<(std::ostream& os, const Result& o);
		};
	}
}

#endif#pragma once
