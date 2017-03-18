#include "InPlaceStableSorter.h"

template<typename T>
inline void sspp::tools::sorters::InPlaceStableSorter::Swap(T & lhs, T & rhs) {
  T tmp = lhs;
  lhs = rhs;
  rhs = tmp;
}

void sspp::tools::sorters::InPlaceStableSorter::Sort(std::vector<INDEXING_TYPE>& I, std::vector<INDEXING_TYPE>& J, std::vector<FLOATING_TYPE>& values, INDEXING_TYPE N) {
  InsertionSort(I, J, values, 0, N);
  auto beginIndex = 0;
  for(auto i = 1; i < N; i++) {
    if(I[i - 1] != I[i]) {
      InsertionSort(J, I, values, beginIndex, i);
      beginIndex = i;
    }
  }
}

void sspp::tools::sorters::InPlaceStableSorter::InsertionSort(std::vector<INDEXING_TYPE>& I, std::vector<INDEXING_TYPE>& J, std::vector<FLOATING_TYPE>& values, INDEXING_TYPE start, INDEXING_TYPE end) {
  for(auto i = start; i < end; i++) {
    auto k = i;
    while(k > start && I[k] < I[k - 1]) {
      Swap(I[k], I[k - 1]);
      Swap(J[k], J[k - 1]);
      Swap(values[k], values[k - 1]);
      k--;
    }
  }
}
