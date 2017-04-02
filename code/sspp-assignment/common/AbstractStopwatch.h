#ifndef SSPP_COMMON_ABSTRACTSTOPWATCH_H_
#define SSPP_COMMON_ABSTRACTSTOPWATCH_H_

namespace sspp {
  namespace common {
    template<typename T>
    class AbstractStopWatch {
    public:
      virtual ~AbstractStopWatch() = default;
      virtual void Start() = 0;
      virtual void Stop() = 0;
      virtual void Reset() = 0;
      virtual double GetElapsedSeconds() const = 0;
    protected:
      T begin_;
      T end_;
    };
  }
}
#endif
