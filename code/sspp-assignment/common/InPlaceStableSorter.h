#ifndef SSPP_COMMON_INPLACESTABLESORTER_H_
#define SSPP_COMMON_INPLACESTABLESORTER_H_

#include "Definitions.h"

namespace sspp {
  namespace tools {
    namespace sorters {
      class InPlaceStableSorter {
      protected:
        template<typename T>
        void swap(T & lhs, T & rhs);
      public:
        void quicksort(int *I, int *J, FLOATING_TYPE *values, const int left, const int right);
        void sort(int *I, int *J, FLOATING_TYPE *values, int N);
        void insertionSort(int *I, int *J, FLOATING_TYPE *values, int start, int end);
        void sort2(int *I, int *J, FLOATING_TYPE *values, int N);
      };
    }
  }
}

#endif