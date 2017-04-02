#include "ChronoStopWatchTest.h"
#include <ChronoStopwatch.h>

#include <thread>

TEST_F(ChronoStopWatchTest, ONE_SECOND) {
  sspp::common::ChronoStopwatch stopwatch;
  
  stopwatch.Start();
  std::this_thread::sleep_for(std::chrono::seconds(1));
  stopwatch.Stop();

  double elapsed = stopwatch.GetElapsedSeconds();

  EXPECT_NEAR(1.0, elapsed, 0.1);
}

TEST_F(ChronoStopWatchTest, HALF_SECOND) {
  sspp::common::ChronoStopwatch stopwatch;

  stopwatch.Start();
  std::this_thread::sleep_for(std::chrono::milliseconds(500));
  stopwatch.Stop();

  double elapsed = stopwatch.GetElapsedSeconds();

  EXPECT_NEAR(0.5, elapsed, 0.1);
}