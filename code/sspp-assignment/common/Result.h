#ifndef SSPP_COMMON_RESULT_H_
#define SSPP_COMMON_RESULT_H_

#include "Definitions.h"
#include "Output.h"
#include "SingleHeader.h"

#include <ostream>
#include <string>
#include <vector>

namespace sspp {
  namespace representations {
    namespace result {
      class Result {
      public:
        single::SingleResult serialResult;
        single::SingleResult parallelResult;
        friend std::ostream& operator <<(std::ostream& os, const Result & result);
      };
    }
  }
}


#endif
