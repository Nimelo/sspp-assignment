#ifndef SSPP_COMMON_OUTPUT_H_
#define SSPP_COMMON_OUTPUT_H_

#include "Definitions.h"
#include <ostream>

namespace sspp {
  namespace representations {
    class Output {
    public:
      FLOATING_TYPE *Values;
      int N;
    public:
      Output();
      Output(int size, FLOATING_TYPE *values);
      Output(const Output & other);
      Output & operator=(Output rhs);
      ~Output();

      friend std::ostream& operator <<(std::ostream& os, const Output& o);
    };
  }
}

#endif
