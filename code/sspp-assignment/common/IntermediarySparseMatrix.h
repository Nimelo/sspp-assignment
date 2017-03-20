#ifndef SSPP_COMMON_INETERMEDIARYSPARSEMATRIX
#define SSPP_COMMON_INETERMEDIARYSPARSEMATRIX

#include "Definitions.h"
#include <vector>

namespace sspp {
  namespace representations {
    class IntermediarySparseMatrix {
    public:
      IntermediarySparseMatrix(const INDEXING_TYPE rows, const INDEXING_TYPE columns, const INDEXING_TYPE non_zeros,
                               std::vector<INDEXING_TYPE> * row_indexes, std::vector<INDEXING_TYPE> * column_indexes,
                               std::vector<FLOATING_TYPE> * values);
      ~IntermediarySparseMatrix();

      INDEXING_TYPE GetRows() const;
      INDEXING_TYPE GetColumns() const;
      INDEXING_TYPE GetNonZeros() const;

      std::vector<INDEXING_TYPE> * GetRowIndexes();
      std::vector<INDEXING_TYPE> * GetColumnIndexes();
      std::vector<FLOATING_TYPE> * GetValues();

    protected:
      INDEXING_TYPE non_zeros_;
      INDEXING_TYPE rows_;
      INDEXING_TYPE columns_;
      std::vector<INDEXING_TYPE> *row_indexes_ = nullptr;
      std::vector<INDEXING_TYPE> *column_indexes_ = nullptr;
      std::vector<FLOATING_TYPE> *values_ = nullptr;
    };
  }
}

#endif
