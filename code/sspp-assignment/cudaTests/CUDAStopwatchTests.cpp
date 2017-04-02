#include "CUDAStopwatchTest.h"
#include "../cuda/CUDAStopwatch.h"
#include <thread>


TEST_F(CUDAStopwatchTest, ONE_SECOND) {
  sspp::cuda::CUDAStopwatch stopwatch;

  stopwatch.Start();
  //TODO: Invoke NVIDIA KERNEL THAT WAITS
  std::this_thread::sleep_for(std::chrono::seconds(1));
  stopwatch.Stop();

  double elapsed = stopwatch.GetElapsedSeconds();

  EXPECT_NEAR(1.0, elapsed, 0.1);
}

TEST_F(CUDAStopwatchTest, HALF_SECOND) {
  sspp::cuda::CUDAStopwatch stopwatch;

  stopwatch.Start();
  //TODO: Invoke NVIDIA KERNEL THAT WAITS
  std::this_thread::sleep_for(std::chrono::milliseconds(500));
  stopwatch.Stop();

  double elapsed = stopwatch.GetElapsedSeconds();

  EXPECT_NEAR(0.5, elapsed, 0.1);
}