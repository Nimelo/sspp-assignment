#ifndef SSPP_COMMON_SINGLERESULT_H_
#define SSPP_COMMON_SINGLERESULT_H_

#include "Definitions.h"
#include "Output.h"

#include <ostream>
#include <string>
#include <vector>

namespace sspp {
  namespace representations {
    namespace result {
      namespace single {
        class SingleResult {
        public:
          Output output;
          std::vector<double> executionTimes;
          friend std::ostream& operator <<(std::ostream& os, const SingleResult& result);
        };
      }
    }
  }
}

#endif
