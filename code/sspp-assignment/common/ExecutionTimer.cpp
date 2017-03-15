#include "ExecutionTimer.h"

std::chrono::milliseconds sspp::tools::measurements::ExecutionTimer::measure(std::function<void(void)> function)
{
	auto t1 = std::chrono::high_resolution_clock::now();
	function();
	auto t2 = std::chrono::high_resolution_clock::now();

	return std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
}