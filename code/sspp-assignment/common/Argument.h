#ifndef SSPP_COMMON_ARGUMENT_H_
#define SSPP_COMMON_ARGUMENT_H_

#include "ArgumentType.h"
#include <string>

namespace sspp {
  namespace common {
    struct Argument {
      std::string name;
      ArgumentType type;
      Argument(std::string name, ArgumentType type)
        : name(name), type(type) {
      }
    };
  }
}

#endif
