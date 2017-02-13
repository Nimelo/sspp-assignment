#ifndef __H_OUTPUT
#define __H_OUTPUT

#include "Definitions.h"

namespace representations
{
	namespace output
	{
		class Output
		{
			protected:
				FLOATING_TYPE *values;
				int size;
			public:
				Output(int size, FLOATING_TYPE *values);
				~Output();
				int getSize() const;
				FLOATING_TYPE * getValues() const;
		};
	}
}

#endif