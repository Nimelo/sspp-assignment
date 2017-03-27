#ifndef SSPP_COMMON_OUTPUT_H_
#define SSPP_COMMON_OUTPUT_H_

#include <ostream>
#include <vector>

namespace sspp {
  namespace common {
    template<typename VALUE_TYPE>
    class Output {
    public:
      Output(std::vector<VALUE_TYPE> & values) :
        values_(values) {
      };

      std::vector<VALUE_TYPE> const & GetValues() const {
        return values_;
      }

      friend std::ostream& operator <<(std::ostream& os, const Output& o) {
        for(typename std::vector<VALUE_TYPE>::iterator it = o.values_.begin(); it != o.values_.end(); ++it)
          os << *it << '\t';
      }
    protected:
      std::vector<VALUE_TYPE> values_;
    };
  }
}

#endif
