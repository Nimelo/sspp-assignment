#ifndef SSPP_COMMON_STABLESORTER_H_
#define SSPP_COMMON_STABLESORTER_H_

#include <vector>

namespace sspp {
  namespace common {
    class StableSorter {
    public:
      template<typename VALUE_TYPE>
      static void InsertionSort(std::vector<unsigned long long>& I,
                                std::vector<unsigned long long>& J,
                                std::vector<VALUE_TYPE>& values,
                                unsigned long long N) {
        unsigned long long start = 0, end = N;
        for(unsigned long long i = start; i < end; i++) {
          auto k = i;
          while(k > start && I[k] < I[k - 1]) {
            Swap(I[k], I[k - 1]);
            Swap(J[k], J[k - 1]);
            Swap(values[k], values[k - 1]);
            k--;
          }
        }
      };
    protected:
      template<typename T>
      static inline void Swap(T & lhs, T & rhs) {
        T tmp = lhs;
        lhs = rhs;
        rhs = tmp;
      }
    };
  }
}
#endif
