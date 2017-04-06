#ifndef SSPP_COMMON_ELLPACKTRANSFORMER_H_
#define SSPP_COMMON_ELLPACKTRANSFORMER_H_
#include "MatrixMarket.h"
#include "ELLPACK.h"
#include <algorithm>

namespace sspp {
  namespace common {
    class ELLPACKTransformer {
    public:
      template<typename VALUE_TYPE>
      static ELLPACK<VALUE_TYPE> transform(MatrixMarket<VALUE_TYPE> & matrix_market) {
        return transform<VALUE_TYPE, VALUE_TYPE>(matrix_market);
      }

      template<typename VALUE_TYPE, typename INPUT_TYPE>
      static ELLPACK<VALUE_TYPE> transform(MatrixMarket<INPUT_TYPE> & matrix_market) {
        std::vector<MatrixMarketTuple<INPUT_TYPE>> tuples = matrix_market.GetTuples();
        std::vector<unsigned> auxiliary_vector(matrix_market.GetRows());
        for(unsigned i = 0; i < matrix_market.GetNonZeros(); i++) {
          ++auxiliary_vector[tuples[i].GetRowIndice()];
        }

        unsigned max_row_non_zeros = *std::max_element(auxiliary_vector.begin(), auxiliary_vector.end());
        std::vector<unsigned> column_indices(max_row_non_zeros * matrix_market.GetRows());
        std::vector<VALUE_TYPE> values(max_row_non_zeros * matrix_market.GetRows());

        for(unsigned i = 0; i < matrix_market.GetNonZeros(); ++i) {
          unsigned index = max_row_non_zeros * tuples[i].GetRowIndice() + auxiliary_vector[tuples[i].GetRowIndice()] - 1;
          column_indices[index] = tuples[i].GetColumnIndice();
          values[index] = static_cast<VALUE_TYPE>(tuples[i].GetValue());
          --auxiliary_vector[tuples[i].GetRowIndice()];
        }

        return ELLPACK<VALUE_TYPE>(matrix_market.GetRows(),
                                   matrix_market.GetColumns(),
                                   matrix_market.GetNonZeros(),
                                   max_row_non_zeros,
                                   column_indices,
                                   values);
      }
    };
  }
}

#endif
