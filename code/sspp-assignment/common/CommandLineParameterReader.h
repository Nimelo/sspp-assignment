#ifndef SSPP_COMMON_COMMANDLINEPARAMETERREADER_H_
#define SSPP_COMMON_COMMANDLINEPARAMETERREADER_H_

#include <vector>
#include "Argument.h"
#include "Parameter.h"

namespace sspp {
  namespace common {
    class CommandLineParameterReader {
    public:
      CommandLineParameterReader(std::vector<Argument> & arguments);
      bool HasArgument(std::string key) const;
      Parameter GetParameter(std::string key) const;
      void Load(const int argc, const char ** argv);
    protected:
      void ExtractSingle(int argc, const char **argv, int index, std::vector<int> & used_indexes, std::string key);
      void ExtractMultiple(int argc, const char **argv, int index, std::vector<int> & used_indexes, std::string key);
      bool HasValue(int index, std::vector<int> & values);

      std::vector<Argument> arguments_;
      std::vector<Parameter> parameters_;
    };
  }
}

#endif
