#include "CommandLineParameterReader.h"
#include "CommandLineException.h"

#include <algorithm>

io::readers::input::commandline::CommandLineParameterReader::CommandLineParameterReader(std::vector<io::readers::input::commandline::arguments::Argument> arguments)
	: arguments(arguments)
{
}

bool io::readers::input::commandline::CommandLineParameterReader::hasArgument(std::string key) const
{
	for (int i = 0; i < parameters.size(); i++)
	{
		if (parameters.at(i).key == key)
		{
			return true;
		}
	}

	return false;
}

io::readers::input::commandline::parameters::Parameter io::readers::input::commandline::CommandLineParameterReader::get(std::string key)
{
	for (int i = 0; i < parameters.size(); i++)
	{
		if (parameters.at(i).key == key)
		{
			return parameters.at(i);
		}
	}

	//return (int)0;
}

void io::readers::input::commandline::CommandLineParameterReader::load(const int argc, const char ** argv)
{
	std::vector<int> usedIndexes;
	for (auto argument : arguments)
	{
		for (int i = 0; i < argc; i++)
		{
			std::string current(argv[i]);

			if (current == argument.Name)
			{
				if (!hasValue(i, usedIndexes))
				{
					usedIndexes.push_back(i);
					if (argument.Type == arguments::ArgumentType::Single)
					{
						extractSingle(argc, argv, i, usedIndexes, argument.Name);
					}
					else
					{
						extractMultiple(argc, argv, i, usedIndexes, argument.Name);
					}
				}
				else
				{
					throw io::exceptions::CommandLineException();
				}
			}
		}
	}
}

void io::readers::input::commandline::CommandLineParameterReader::extractSingle(int argc, const char ** argv, int index, std::vector<int>& usedIndexes, std::string key)
{
	if (index < argc)
	{
		if (!hasValue(index + 1, usedIndexes))
		{
			parameters.push_back(parameters::Parameter(key, argv[index + 1]));
			usedIndexes.push_back(index + 1);
			return;
		}
	}

	throw io::exceptions::CommandLineException();
}

void io::readers::input::commandline::CommandLineParameterReader::extractMultiple(int argc, const char ** argv, int index, std::vector<int>& usedIndexes, std::string key)
{
	if (index < argc)
	{
		if (!hasValue(index + 1, usedIndexes))
		{
			std::vector<std::string> values;
			for (int i = index + 1; i < argc; i++)
			{
				if (!hasValue(i, usedIndexes))
				{
					usedIndexes.push_back(i);
					values.push_back(argv[i]);
				}
				else
				{
					break;
				}
			}

			parameters.push_back(parameters::Parameter(key, values));
			return;
		}
	}

	throw io::exceptions::CommandLineException();
}

bool io::readers::input::commandline::CommandLineParameterReader::hasValue(int index, std::vector<int>& values)
{
	for (int v : values)
	{
		if (v == index)
			return true;
	}

	return false;
}
