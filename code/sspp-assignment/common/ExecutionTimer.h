#ifndef SSPP_COMMON_EXECUTIONTIMER_H_
#define SSPP_COMMON_EXECUTIONTIMER_H_

#include <chrono>
#include <functional>

namespace sspp {
  namespace tools {
    namespace measurements {
      class ExecutionTimer {
      public:
        std::chrono::milliseconds measure(std::function<void(void)> function);
      };
    }
  }
}

#endif