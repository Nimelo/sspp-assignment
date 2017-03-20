#ifndef SSPP_COMMON_OUTPUT_H_
#define SSPP_COMMON_OUTPUT_H_

#include "Definitions.h"
#include <ostream>
#include <vector>

namespace sspp {
  namespace representations {
    class Output {
    public:
      Output();
      ~Output();
      Output(std::vector<FLOATING_TYPE> * values);    
      std::vector<FLOATING_TYPE> *GetValues() const;
      friend std::ostream& operator <<(std::ostream& os, const Output& o);
    protected:
      std::vector<FLOATING_TYPE> *values_;
    };
  }
}

#endif
