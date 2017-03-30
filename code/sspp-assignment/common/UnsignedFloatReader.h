#ifndef SSPP_COMMON_UNSIGNEDFLOATREADER_H_
#define SSPP_COMMON_UNSIGNEDFLOATREADER_H_
#include "MatrixMarketTupleReaderInterface.h"

#include <string>
#include <sstream>

namespace sspp {
  namespace common {
    class UnsignedFlaotReader : public MatrixMarketTupleReaderInterface<float> {
    public:
      MatrixMarketTuple<float> Get(std::string line, bool is_pattern) {
        std::stringstream ss(line);

        unsigned row_indice, column_indice;
        float value;
        ss >> row_indice >> column_indice;
        if(is_pattern) {
          value = GetPatternValue();
        } else {
          ss >> value;
        }

        return MatrixMarketTuple<float>(row_indice - 1, column_indice - 1, value);
      };
    public:
      float GetPatternValue() const {
        return static_cast<float>(rand() % 100);
      }
    };
  }
}
#endif
