#include "ExecutionTimer.h"

std::chrono::milliseconds tools::measurements::timers::ExecutionTimer::measure(std::function<void(void)> function)
{
	std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
	function();
	std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();

	return std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
}