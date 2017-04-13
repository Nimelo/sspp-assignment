#include "CommandLineParameterReader.h"
#include "CommandLineParameterReader.h"
#include "CommandLineException.h"

#include <algorithm>
namespace sspp {
  namespace common {
    CommandLineParameterReader::CommandLineParameterReader(std::vector<Argument> & arguments)
      : arguments_(arguments) {
    }

    bool CommandLineParameterReader::HasArgument(std::string key) const {
      for(unsigned long long int i = 0; i < parameters_.size(); i++) {
        if(parameters_.at(i).GetKey() == key) {
          return true;
        }
      }

      return false;
    }

    Parameter CommandLineParameterReader::GetParameter(std::string key) const {
      for(unsigned long long int i = 0; i < parameters_.size(); i++) {
        if(parameters_.at(i).GetKey() == key) {
          return parameters_.at(i);
        }
      }

      throw CommandLineException();
    }

    void CommandLineParameterReader::Load(const int argc, const char ** argv) {
      std::vector<int> used_indexes;
      for(auto argument : arguments_) {
        for(auto i = 0; i < argc; i++) {
          std::string current_argument_value(argv[i]);

          if(current_argument_value == argument.name) {
            if(!HasValue(i, used_indexes)) {
              used_indexes.push_back(i);
              if(argument.type == ArgumentType::Flag) {
                this->parameters_.push_back(Parameter(argument.name));
              } else if(argument.type == ArgumentType::Single) {
                ExtractSingle(argc, argv, i, used_indexes, argument.name);
              } else {
                ExtractMultiple(argc, argv, i, used_indexes, argument.name);
              }
            } else {
              throw CommandLineException();
            }
          }
        }
      }
    }

    void CommandLineParameterReader::ExtractSingle(int argc, const char ** argv, int index, std::vector<int>& usedIndexes, std::string key) {
      if(index < argc) {
        if(!HasValue(index + 1, usedIndexes)) {
          parameters_.push_back(Parameter(key, argv[index + 1]));
          usedIndexes.push_back(index + 1);
          return;
        }
      }

      throw CommandLineException();
    }

    void CommandLineParameterReader::ExtractMultiple(int argc, const char ** argv, int index, std::vector<int>& used_indexes, std::string key) {
      if(index < argc) {
        if(!HasValue(index + 1, used_indexes)) {
          std::vector<std::string> values;
          for(auto i = index + 1; i < argc; i++) {
            if(!HasValue(i, used_indexes)) {
              used_indexes.push_back(i);
              values.push_back(argv[i]);
            } else {
              break;
            }
          }

          parameters_.push_back(Parameter(key, values));
          return;
        }
      }

      throw CommandLineException();
    }

    bool CommandLineParameterReader::HasValue(int index, std::vector<int>& values) {
      for(auto v : values) {
        if(v == index)
          return true;
      }

      return false;
    }
  }
}