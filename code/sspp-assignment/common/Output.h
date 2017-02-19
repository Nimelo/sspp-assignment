#ifndef __H_OUTPUT
#define __H_OUTPUT

#include "Definitions.h"
#include <ostream>

namespace representations
{
	namespace output
	{
		class Output
		{
			public:
				FLOATING_TYPE *Values;
				int N;
			public:
				Output(int size, FLOATING_TYPE *values);
				Output(const Output & other);
				~Output();

				friend std::ostream& operator <<(std::ostream& os, const Output& o);
		};
	}
}

#endif