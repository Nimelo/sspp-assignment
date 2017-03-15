#ifndef SSPP_COMMON_ARGUMENT_H_
#define SSPP_COMMON_ARGUMENT_H_

#include "ArgumentType.h"
#include <string>

namespace sspp {
  namespace io {
    namespace readers {
      namespace commandline {
        struct Argument {
          std::string Name;
          ArgumentType Type;
          Argument(std::string name, ArgumentType type)
            : Name(name), Type(type) {
          }
        };
      }
    }
  }
}

#endif
