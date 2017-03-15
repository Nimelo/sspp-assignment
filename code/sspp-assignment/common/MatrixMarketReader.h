#ifndef SSPP_COMMON_MATRIXMARKETREADER_H_
#define SSPP_COMMON_MATRIXMARKETREADER_H_

#include "IntermediarySparseMatrix.h"
#include <string>

namespace sspp {
  namespace io {
    namespace readers {
      class MatrixMarketReader {
      public:
        representations::IntermediarySparseMatrix fromFile(std::string file_name);
      };
    }
  }
}

#endif
