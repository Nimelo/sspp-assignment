#ifndef SSPP_COMMON_ARGUMENTTYPE_H_
#define SSPP_COMMON_ARGUMENTTYPE_H_

namespace sspp {
  namespace io {
    namespace readers {
      namespace commandline {
        enum ArgumentType {
          Flag,
          Single,
          Multiple
        };
      }
    }
  }
}

#endif
