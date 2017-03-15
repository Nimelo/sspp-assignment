#ifndef SSPP_COMMON_COMMANDLINEPARAMETERREADER_H_
#define SSPP_COMMON_COMMANDLINEPARAMETERREADER_H_

#include <vector>
#include "Argument.h"
#include "Parameter.h"

namespace sspp {
  namespace io {
    namespace readers {
      namespace commandline {
        class CommandLineParameterReader {
        public:
          CommandLineParameterReader(std::vector<Argument> arguments);
          bool hasArgument(std::string key) const;
          Parameter get(std::string key);
          void load(const int argc, const char ** argv);
        protected:
          std::vector<Argument> arguments;
          std::vector<Parameter> parameters;
          void extractSingle(int argc, const char **argv, int index, std::vector<int> & used_indexes, std::string key);
          void extractMultiple(int argc, const char **argv, int index, std::vector<int> & used_indexes, std::string key);
          bool hasValue(int index, std::vector<int> & values);
        };
      }
    }
  }
}

#endif
