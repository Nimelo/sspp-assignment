#ifndef SSPP_COMMON_PARAMETER_H_
#define SSPP_COMMON_PARAMETER_H_

#include "ConversionFailedException.h"

#include <vector>
#include <string>
#include <sstream>

namespace sspp {
  namespace io {
    namespace readers {
      namespace commandline {
        class Parameter {
        public:
          Parameter(std::string key)
            : key_(key) {

          }

          Parameter(std::string key, std::string value)
            : value_(value), key_(key) {
            values_list_.push_back(value);
          }

          Parameter(std::string key, std::vector<std::string> list)
            : values_list_(list), key_(key) {
            if(!list.empty())
              value_ = list.at(0);
          }

          template<typename T>
          operator T() const {
            std::stringstream ss(value_);
            T converted_value;
            if(ss >> converted_value) return converted_value;
            else throw io::exceptions::ConversionFailedException();
          }

          template<typename T>
          operator std::vector<T>() const {
            std::vector<T> return_list;

            for(auto str : values_list_) {
              std::stringstream ss(str);
              T converted_value;
              if(ss >> converted_value)
                return_list.push_back(converted_value);
              else throw io::exceptions::ConversionFailedException();
            }

            return return_list;
          }

          std::string GetKey() const {
            return this->key_;
          }
        protected:
          std::string key_;
          std::string value_;
          std::vector<std::string> values_list_;
        };
      }
    }
  }
}

#endif
