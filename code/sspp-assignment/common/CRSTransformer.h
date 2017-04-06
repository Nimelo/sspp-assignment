#ifndef SSPP_COMMON_CRSTRANSFORMER_H_
#define SSPP_COMMON_CRSTRANSFORMER_H_
#include "CRS.h"
#include "MatrixMarket.h"

#include <algorithm>

namespace sspp {
  namespace common {
    class CRSTransformer {
    public:
      template<typename VALUE_TYPE>
      static CRS<VALUE_TYPE> transform(MatrixMarket<VALUE_TYPE> & matrix_market) {
        return transform<VALUE_TYPE, VALUE_TYPE>(matrix_market);
      }
      template<typename VALUE_TYPE, typename INPUT_VALUE>
      static CRS<VALUE_TYPE> transform(MatrixMarket<INPUT_VALUE> & matrix_market) {
        std::vector<MatrixMarketTuple<INPUT_VALUE>> tuples = matrix_market.GetTuples();
        std::sort(tuples.begin(), tuples.end(), MatrixMarket<INPUT_VALUE>::CompareMatrixMarketTuple);
        std::vector<unsigned> column_indices(matrix_market.GetNonZeros()), row_start_indexes(matrix_market.GetRows() + 1);
        std::vector<VALUE_TYPE> values(matrix_market.GetNonZeros());

        unsigned row_start_indexes_index = 0, non_zeros_index = 1;
        row_start_indexes[row_start_indexes_index++] = 0;

        while(row_start_indexes_index < matrix_market.GetRows() + 1
              && non_zeros_index < matrix_market.GetNonZeros()) {
          column_indices[non_zeros_index - 1] = tuples[non_zeros_index - 1].GetColumnIndice();
          values[non_zeros_index - 1] = static_cast<VALUE_TYPE>(tuples[non_zeros_index - 1].GetValue());
          unsigned row_indices_diff = tuples[non_zeros_index].GetRowIndice() - tuples[non_zeros_index - 1].GetRowIndice();
          if(row_indices_diff != 0) {
            row_start_indexes[row_start_indexes_index++] = non_zeros_index;
            if(row_indices_diff > 1) {
              for(unsigned i = 0; i < row_indices_diff - 1; ++i) {
                row_start_indexes[row_start_indexes_index++] = non_zeros_index;
              }
            }
          }
          ++non_zeros_index;
        }
        for(unsigned i = row_start_indexes_index; i < matrix_market.GetRows(); ++i) {
          row_start_indexes[row_start_indexes_index++] = non_zeros_index;
        }
        row_start_indexes[row_start_indexes_index++] = non_zeros_index;
        column_indices[non_zeros_index - 1] = tuples[non_zeros_index - 1].GetColumnIndice();
        values[non_zeros_index - 1] = static_cast<VALUE_TYPE>(tuples[non_zeros_index - 1].GetValue());

        return CRS<VALUE_TYPE>(matrix_market.GetRows(),
                               matrix_market.GetColumns(),
                               matrix_market.GetNonZeros(),
                               row_start_indexes,
                               column_indices,
                               values);
      }
    };
  }
}
#endif
