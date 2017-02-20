#ifndef __H_COMMAND_LINE_PARAMETER_READER
#define __H_COMMAND_LINE_PARAMETER_READER

#include <vector>
#include "Argument.h"
#include "Parameter.h"

namespace io
{
	namespace readers
	{
		namespace input
		{
			namespace commandline
			{
				class CommandLineParameterReader
				{
					protected:
						std::vector<io::readers::input::commandline::arguments::Argument> arguments;
						std::vector<io::readers::input::commandline::parameters::Parameter> parameters;
						void extractSingle(int argc, const char **argv, int index, std::vector<int> & usedIndexes, std::string key);
						void extractMultiple(int argc, const char **argv, int index, std::vector<int> & usedIndexes, std::string key);
						bool hasValue(int index, std::vector<int> & values);
					public:
						CommandLineParameterReader(std::vector<io::readers::input::commandline::arguments::Argument> arguments);
						bool hasArgument(std::string key) const;
						parameters::Parameter get(std::string key);
						void load(const int argc, const char ** argv);
				};
			}
		}
	}
}

#endif