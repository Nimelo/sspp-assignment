#ifndef __H_OUTPUT
#define __H_OUTPUT

#include "Definitions.h"

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
		};
	}
}

#endif