#ifndef SSPP_COMMON_INETERMEDIARYSPARSEMATRIX
#define SSPP_COMMON_INETERMEDIARYSPARSEMATRIX

#include "Definitions.h"
#include <vector>

namespace sspp {
  namespace representations {
    class IntermediarySparseMatrix {
    public:
      IntermediarySparseMatrix() = default;
      ~IntermediarySparseMatrix() = default;

      IntermediarySparseMatrix(const INDEXING_TYPE rows, const INDEXING_TYPE columns, const INDEXING_TYPE non_zeros,
                               const std::vector<INDEXING_TYPE> & row_indexes, const std::vector<INDEXING_TYPE> & column_indexes,
                               const std::vector<FLOATING_TYPE> & values);
      IntermediarySparseMatrix(const IntermediarySparseMatrix &other);
      IntermediarySparseMatrix & operator=(const IntermediarySparseMatrix & rhs);

      INDEXING_TYPE GetRows() const;
      INDEXING_TYPE GetColumns() const;
      INDEXING_TYPE GetNonZeros() const;

      std::vector<INDEXING_TYPE> GetRowIndexes() const;
      std::vector<INDEXING_TYPE> GetColumnIndexes() const;
      std::vector<FLOATING_TYPE> GetValues() const;

    protected:
      static void Rewrite(IntermediarySparseMatrix & lhs, const IntermediarySparseMatrix & rhs);
      INDEXING_TYPE non_zeros_;
      INDEXING_TYPE rows_;
      INDEXING_TYPE columns_;
      std::vector<INDEXING_TYPE> row_indexes_;
      std::vector<INDEXING_TYPE> column_indexes_;
      std::vector<FLOATING_TYPE> values_;
    };
  }
}

#endif
