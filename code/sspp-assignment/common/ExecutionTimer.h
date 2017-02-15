#ifndef __H_EXECUTION_TIMER
#define __H_EXECUTION_TIMER

#include <chrono>
#include <functional>

namespace tools
{
	namespace measurements
	{
		namespace timers
		{
			class ExecutionTimer
			{
				public:
					std::chrono::milliseconds measure(std::function<void(void)> function);
			};
		}
	}
}

#endif