#ifndef SSPP_COMMON_ABSTRACTSTOPWATCH_H_
#define SSPP_COMMON_ABSTRACTSTOPWATCH_H_

namespace sspp {
  namespace common {
    class AbstractStopWatch {
    public:
      ~AbstractStopWatch() = default;
      virtual void Start() = 0;
      virtual void Stop() = 0;
      virtual void Reset() = 0;
      virtual double GetElapsedSeconds() const = 0;
    };
  }
}
#endif
