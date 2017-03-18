#ifndef SSPP_COMMON_MATRIXMARKETREADER_H_
#define SSPP_COMMON_MATRIXMARKETREADER_H_

#include "IntermediarySparseMatrix.h"
#include <string>
#include <istream>

namespace sspp {
  namespace io {
    namespace readers {
      class MatrixMarketReader {
      public:
        representations::IntermediarySparseMatrix FromFile(std::string file_name);
        representations::IntermediarySparseMatrix FromStream(std::istream & is);
      };
    }
  }
}

#endif
