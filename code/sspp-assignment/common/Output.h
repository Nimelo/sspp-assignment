#ifndef SSPP_COMMON_OUTPUT_H_
#define SSPP_COMMON_OUTPUT_H_

#include <ostream>
#include <vector>

namespace sspp {
  namespace common {
    template<typename VALUE_TYPE>
    class Output {
    public:
      Output() {};
      Output(std::vector<VALUE_TYPE> & values) {
        values_.assign(values.begin(), values.end());
        seconds_ = 0;
      };

      Output(std::vector<VALUE_TYPE> & values, double ms) {
        values_.assign(values.begin(), values.end());
        seconds_ = ms;
      };

      Output(const Output<VALUE_TYPE> & other) {
        Swap(*this, other);
      }

      Output<VALUE_TYPE> & operator=(const Output<VALUE_TYPE> & other) {
        Swap(*this, other);
        return *this;
      }

      std::vector<VALUE_TYPE> const & GetValues() const {
        return values_;
      }

      double GetSeconds() const {
        return seconds_;
      }

      unsigned SetSeconds(double ms) {
        seconds_ = ms;
      }

      friend std::ostream& operator <<(std::ostream& os, const Output<VALUE_TYPE>& o) {
        for(unsigned i = 0; i < o.GetValues().size(); ++i)
          os << o.GetValues()[i] << ' ';
        return os;
      }
    protected:
      void Swap(Output & lhs, const Output & rhs) {
        lhs.values_.resize(rhs.values_.size());
        if(!rhs.values_.empty())
          copy(rhs.values_.begin(), rhs.values_.end(), lhs.values_.begin());

        lhs.seconds_ = rhs.seconds_;
      }

      std::vector<VALUE_TYPE> values_;
      double seconds_;
    };
  }
}

#endif
