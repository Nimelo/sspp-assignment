#ifndef SSPP_COMMON_MATRIXMARKET_H_
#define SSPP_COMMON_MATRIXMARKET_H_

#include <vector>
#include "MatrixMarketTuple.h"
#include "MatrixMarketHeader.h"
#include "SparseMatrix.h"

namespace sspp {
  namespace common {
    template<typename VALUE_TYPE>
    class MatrixMarket : public SparseMatrix {
    public:
      MatrixMarket(const unsigned rows,
                   const unsigned columns,
                   const unsigned non_zeros,
                   const std::vector<unsigned> & row_indices,
                   const std::vector<unsigned> & column_indices,
                   const std::vector<VALUE_TYPE> & values)
        : SparseMatrix(rows, columns, non_zeros),
        row_indices_(row_indices),
        column_indices_(column_indices),
        values_(values) {
        tuples_.resize(non_zeros_);
        for(unsigned i = 0; i < tuples_.size(); i++) {
          tuples_[i] = MatrixMarketTuple<VALUE_TYPE>(row_indices_[i], column_indices_[i], values_[i]);
        }
      };

      std::vector<MatrixMarketTuple<VALUE_TYPE>> GetTuples() const {
        return tuples_;
      }

      std::vector<unsigned> const & GetRowIndices() const {
        return row_indices_;
      }

      std::vector<unsigned> const & GetColumnIndices() const {
        return column_indices_;
      }

      std::vector<VALUE_TYPE> const & GetValues() const {
        return values_;
      }

      static bool CompareMatrixMarketTuple(MatrixMarketTuple<VALUE_TYPE> & lhs, MatrixMarketTuple<VALUE_TYPE> & rhs) {
        return ((lhs.GetRowIndice() < rhs.GetRowIndice()) ||
          (lhs.GetRowIndice() == rhs.GetRowIndice() && lhs.GetColumnIndice() < rhs.GetColumnIndice()));
      }

    private:
      std::vector<unsigned> row_indices_;
      std::vector<unsigned> column_indices_;
      std::vector<VALUE_TYPE> values_;
      std::vector<MatrixMarketTuple<VALUE_TYPE>> tuples_;
    };
  }
}

#endif
