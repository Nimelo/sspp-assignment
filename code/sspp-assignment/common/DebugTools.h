#ifndef __H_DEBUG_TOOLS
#define __H_DEBUG_TOOLS

#include <iostream>

namespace tools
{
	namespace debug
	{
		class DebugTools
		{
		public:
			template<typename T>
			void printArray(T original, int n, const char * str)
			{
				std::cout << str << ": [";
				for (int i = 0; i < n; i++)
				{
					std::cout << original[i] << ", ";
				}
				std::cout << "]\n";
			}
		};
	}
}

#endif