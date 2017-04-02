#include "OpenMPStopwatchTest.h"

#include "OpenMPStopwatch.h"
#include <thread>

TEST_F(OpenMPStopwatchTest, ONE_SECOND) {
  sspp::openmp::OpenMPStopwatch stopwatch;

  stopwatch.Start();
  std::this_thread::sleep_for(std::chrono::seconds(1));
  stopwatch.Stop();

  double elapsed = stopwatch.GetElapsedSeconds();

  EXPECT_NEAR(1.0, elapsed, 0.1);
}

TEST_F(OpenMPStopwatchTest, HALF_SECOND) {
  sspp::openmp::OpenMPStopwatch stopwatch;

  stopwatch.Start();
  std::this_thread::sleep_for(std::chrono::milliseconds(500));
  stopwatch.Stop();

  double elapsed = stopwatch.GetElapsedSeconds();

  EXPECT_NEAR(0.5, elapsed, 0.1);
}