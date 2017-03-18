#ifndef SSPP_COMMON_OUTPUT_H_
#define SSPP_COMMON_OUTPUT_H_

#include "Definitions.h"
#include <ostream>
#include <vector>

namespace sspp {
  namespace representations {
    class Output {
    public:
      Output() = default;
      ~Output() = default;

      Output(std::vector<FLOATING_TYPE> & values);
      Output(const Output & other);
      Output & operator=(const Output & rhs); 
      
      std::vector<FLOATING_TYPE> GetValues() const;

      friend std::ostream& operator <<(std::ostream& os, const Output& o);
    protected:
      static void Rewrite(Output & lhs, const Output & rhs);
      std::vector<FLOATING_TYPE> values_;
    };
  }
}

#endif
