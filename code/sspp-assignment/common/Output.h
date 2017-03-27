#ifndef SSPP_COMMON_OUTPUT_H_
#define SSPP_COMMON_OUTPUT_H_

#include <ostream>
#include <vector>

namespace sspp {
  namespace common {
    template<typename VALUE_TYPE>
    class Output {
    public:
      Output(std::vector<VALUE_TYPE> & values) {
        values_.assign(values.begin(), values.end());
      };

      Output(const Output<VALUE_TYPE> & other) {
        Swap(*this, other);
      }

      Output<VALUE_TYPE> & operator=(const Output<VALUE_TYPE> other) {
        Swap(*this, other);
        return *this;
      }

      std::vector<VALUE_TYPE> const & GetValues() const {
        return values_;
      }

      friend std::ostream& operator <<(std::ostream& os, const Output& o) {
        for(typename std::vector<VALUE_TYPE>::iterator it = o.values_.begin(); it != o.values_.end(); ++it)
          os << *it << '\t';
      }
    protected:
      void Swap(Output & lhs, const Output & rhs) {
        lhs.values_.resize(rhs.values_.size());
        if(!rhs.values_.empty())
        copy(rhs.values_.begin(), rhs.values_.end(), lhs.values_.begin());
        //lhs.values_.assign(rhs.va, lhs.values_.end());
      }

      std::vector<VALUE_TYPE> values_;
    };
  }
}

#endif
