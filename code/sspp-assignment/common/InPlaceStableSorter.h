#ifndef SSPP_COMMON_INPLACESTABLESORTER_H_
#define SSPP_COMMON_INPLACESTABLESORTER_H_

#include "Definitions.h"

namespace sspp {
  namespace tools {
    namespace sorters {
      class InPlaceStableSorter {
      protected:
        template<typename T>
        void Swap(T & lhs, T & rhs);
      public:
        void Sort(INDEXING_TYPE *I, INDEXING_TYPE *J, FLOATING_TYPE *values, INDEXING_TYPE N);
        void InsertionSort(INDEXING_TYPE *I, INDEXING_TYPE *J, FLOATING_TYPE *values, INDEXING_TYPE start, INDEXING_TYPE end);
      };
    }
  }
}

#endif