#ifndef __IN_PLACE_SORTER
#define __IN_PLACE_SORTER

#include "Definitions.h"

namespace tools
{
	namespace sorters
	{
		class InPlaceSorter
		{
			protected:
				template<typename T>
				void swap(T & lhs, T & rhs);
				template<typename T>
				void quicksort(int *I, int *J, FLOATING_TYPE *values, const T left, const T right);
			public:
				void sort(int *I, int *J, FLOATING_TYPE *values, int N);
		};
	}
}

#endif