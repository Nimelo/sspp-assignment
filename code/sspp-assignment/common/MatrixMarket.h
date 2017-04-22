#ifndef SSPP_COMMON_MATRIXMARKET_H_
#define SSPP_COMMON_MATRIXMARKET_H_

#include <vector>
#include "MatrixMarketTuple.h"
#include "MatrixMarketHeader.h"
#include "SparseMatrix.h"
#include "MatrixMarketInfo.h"
#include <numeric>

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

      static bool CompareMatrixMarketTuple(const MatrixMarketTuple<VALUE_TYPE> & lhs, const MatrixMarketTuple<VALUE_TYPE> & rhs) {
        return ((lhs.GetRowIndice() < rhs.GetRowIndice()) ||
          (lhs.GetRowIndice() == rhs.GetRowIndice() && lhs.GetColumnIndice() < rhs.GetColumnIndice()));
      }

      MatrixMarketInfo GetInfo() {
        std::vector<unsigned> auxiliary_array(rows_);

        for(auto const & tuple : tuples_) {
          ++auxiliary_array[tuple.GetRowIndice()];
        }

        double max = *std::max_element(auxiliary_array.begin(), auxiliary_array.end());
        double min = *std::min_element(auxiliary_array.begin(), auxiliary_array.end());
        double average = std::accumulate(auxiliary_array.begin(), auxiliary_array.end(), 0.0) / auxiliary_array.size();

        double standard_deviation = 0.0;
        for(auto const & val : auxiliary_array) {
          standard_deviation += (val - average) * (val - average);
        }

        standard_deviation /= auxiliary_array.size();

        return MatrixMarketInfo(rows_, columns_, non_zeros_, max, min, average, standard_deviation);

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
