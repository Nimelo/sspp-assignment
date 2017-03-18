#pragma once

#include "MatrixMarketHeader.h"
#include "MatrixMarketEnums.h"
#include "Definitions.h"
#include <istream>
#include <vector>

namespace sspp {
  namespace io {
    class MatrixMarket {
    public:
      MatrixMarket(MatrixMarketHeader header);
      MatrixMarketErrorCodes ReadIndices(std::istream & is,
                                         INDEXING_TYPE & rows, INDEXING_TYPE & columns, INDEXING_TYPE & non_zeros,
                                         std::vector<INDEXING_TYPE> & row_indices,
                                         std::vector<INDEXING_TYPE> & column_indices, std::vector<FLOATING_TYPE> & values);
      MatrixMarketErrorCodes MatrixMarket::ReadIndices(std::istream &is,
                                                       INDEXING_TYPE & non_zeros,
                                                       std::vector<INDEXING_TYPE>& row_indices,
                                                       std::vector<INDEXING_TYPE>& column_indices,
                                                       std::vector<FLOATING_TYPE>& values);
    private:
      static constexpr const char COMMENT_PREFIX_STR[] = "%";
      static constexpr const int RAND_LIMIT = 100;
      MatrixMarketHeader header_;
    };
  }
}