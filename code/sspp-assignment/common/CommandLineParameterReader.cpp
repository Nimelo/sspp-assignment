#include "CommandLineParameterReader.h"
#include "CommandLineException.h"

#include <algorithm>

sspp::io::readers::commandline::CommandLineParameterReader::CommandLineParameterReader(std::vector<sspp::io::readers::commandline::Argument> arguments)
  : arguments(arguments) {
}

bool sspp::io::readers::commandline::CommandLineParameterReader::hasArgument(std::string key) const {
  for(auto i = 0; i < parameters.size(); i++) {
    if(parameters.at(i).key == key) {
      return true;
    }
  }

  return false;
}

sspp::io::readers::commandline::Parameter sspp::io::readers::commandline::CommandLineParameterReader::get(std::string key) {
  for(auto i = 0; i < parameters.size(); i++) {
    if(parameters.at(i).key == key) {
      return parameters.at(i);
    }
  }

  throw sspp::io::exceptions::CommandLineException();
}

void sspp::io::readers::commandline::CommandLineParameterReader::load(const int argc, const char ** argv) {
  std::vector<int> usedIndexes;
  for(auto argument : arguments) {
    for(auto i = 0; i < argc; i++) {
      std::string current(argv[i]);

      if(current == argument.Name) {
        if(!hasValue(i, usedIndexes)) {
          usedIndexes.push_back(i);
          if(argument.Type == ArgumentType::Flag) {
            this->parameters.push_back(Parameter(argument.Name));
          } else if(argument.Type == ArgumentType::Single) {
            extractSingle(argc, argv, i, usedIndexes, argument.Name);
          } else {
            extractMultiple(argc, argv, i, usedIndexes, argument.Name);
          }
        } else {
          throw io::exceptions::CommandLineException();
        }
      }
    }
  }
}

void sspp::io::readers::commandline::CommandLineParameterReader::extractSingle(int argc, const char ** argv, int index, std::vector<int>& usedIndexes, std::string key) {
  if(index < argc) {
    if(!hasValue(index + 1, usedIndexes)) {
      parameters.push_back(Parameter(key, argv[index + 1]));
      usedIndexes.push_back(index + 1);
      return;
    }
  }

  throw io::exceptions::CommandLineException();
}

void sspp::io::readers::commandline::CommandLineParameterReader::extractMultiple(int argc, const char ** argv, int index, std::vector<int>& usedIndexes, std::string key) {
  if(index < argc) {
    if(!hasValue(index + 1, usedIndexes)) {
      std::vector<std::string> values;
      for(auto i = index + 1; i < argc; i++) {
        if(!hasValue(i, usedIndexes)) {
          usedIndexes.push_back(i);
          values.push_back(argv[i]);
        } else {
          break;
        }
      }

      parameters.push_back(Parameter(key, values));
      return;
    }
  }

  throw io::exceptions::CommandLineException();
}

bool sspp::io::readers::commandline::CommandLineParameterReader::hasValue(int index, std::vector<int>& values) {
  for(auto v : values) {
    if(v == index)
      return true;
  }

  return false;
}
