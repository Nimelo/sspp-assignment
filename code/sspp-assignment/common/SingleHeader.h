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
          SingleResult();
          ~SingleResult();
          Output *GetOutput() const;
          void SetOutput(Output *output);
          std::vector<double> GetExecutionTimes() const;
          friend std::ostream& operator <<(std::ostream& os, const SingleResult& result);
        private:
          Output *output_;
          std::vector<double> execution_times_;
        };
      }
    }
  }
}

#endif
