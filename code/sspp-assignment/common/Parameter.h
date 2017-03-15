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
          std::string key;
          Parameter(std::string key)
            : key(key) {

          }

          Parameter(std::string key, std::string value)
            : value(value), key(key) {
            valuesList.push_back(value);
          }

          Parameter(std::string key, std::vector<std::string> list)
            : valuesList(list), key(key) {
            if(!list.empty())
              value = list.at(0);
          }

          template<typename T>
          operator T() const {
            std::stringstream ss(value);
            T convertedValue;
            if(ss >> convertedValue) return convertedValue;
            else throw io::exceptions::ConversionFailedException();
          }

          template<typename T>
          operator std::vector<T>() const {
            std::vector<T> returnList;

            for(auto str : valuesList) {
              std::stringstream ss(str);
              T convertedValue;
              if(ss >> convertedValue)
                returnList.push_back(convertedValue);
              else throw io::exceptions::ConversionFailedException();
            }

            return returnList;
          }
        protected:
          std::string value;
          std::vector<std::string> valuesList;
        };
      }
    }
  }
}

#endif
