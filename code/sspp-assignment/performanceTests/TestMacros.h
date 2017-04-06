#pragma once
#include "MatrixTestInterface.h"

#define CREATE_FIXTURE(FIXTURE_NAME, KEY, PATH)   \
class FIXTURE_NAME : public MatrixTestInterface { \
protected:                                        \
   std::string GetPath() {                        \
    return PATH;                                  \
  };                                              \
  std::string GetKey() {                          \
    return KEY;                                   \
  };                                              \
};                                                \
std::string FIXTURE_NAME::static_key_ = KEY;      \
;