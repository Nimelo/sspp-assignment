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
        Result();
        ~Result();
        single::SingleResult *GetSerial() const;
        single::SingleResult *GetParallel() const;
        friend std::ostream& operator <<(std::ostream& os, const Result & result);
      private:
        single::SingleResult *serial_result_;
        single::SingleResult *parallel_result_;
      };
    }
  }
}


#endif
