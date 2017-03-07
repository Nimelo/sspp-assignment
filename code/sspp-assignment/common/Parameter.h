#ifndef __H_PARAMETER
#define __H_PARAMETER

#include <string>
#include <sstream>

namespace io
{
	namespace readers
	{
		namespace input
		{
			namespace commandline
			{
				namespace parameters
				{
					class Parameter
					{
					protected:
						std::string value;
						std::vector<std::string> list;
					public:
						std::string key;
						Parameter(std::string key, std::string value)
							: value(value), key(key)
						{
							list.push_back(value);
						}

						Parameter(std::string key, std::vector<std::string> list)
							: list(list), key(key)
						{
							if (!list.empty())
								value = list.at(0);
						}

						template<typename T> operator T() const
						{
							std::stringstream ss(value);
							T convertedValue;
							if (ss >> convertedValue) return convertedValue;
							else throw std::runtime_error("Conversion failed");
						}

						template<typename T>
						operator std::vector<T>() const
						{
							std::vector<T> returnList;

							for (auto str : list)
							{
								std::stringstream ss(str);
								T convertedValue;
								if (ss >> convertedValue)
									returnList.push_back(convertedValue);
								else
									throw std::runtime_error("Conversion failed");
							}

							return returnList;
						}
					};
				}
			}
		}
	}
}

#endif