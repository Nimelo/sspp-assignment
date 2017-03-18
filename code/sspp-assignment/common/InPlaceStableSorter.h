#ifndef SSPP_COMMON_INPLACESTABLESORTER_H_
#define SSPP_COMMON_INPLACESTABLESORTER_H_

#include "Definitions.h"
#include <vector>

namespace sspp {
  namespace tools {
    namespace sorters {
      class InPlaceStableSorter {
      protected:
        template<typename T>
        void Swap(T & lhs, T & rhs);
      public:
        void Sort(std::vector<INDEXING_TYPE> & I, std::vector<INDEXING_TYPE> & J, std::vector<FLOATING_TYPE> & values, INDEXING_TYPE N);
        void InsertionSort(std::vector<INDEXING_TYPE> & I, std::vector<INDEXING_TYPE> & J, std::vector<FLOATING_TYPE> & values, INDEXING_TYPE start, INDEXING_TYPE end);
      };
    }
  }
}

#endif