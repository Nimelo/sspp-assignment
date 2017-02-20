#ifndef __H_ARGUMENT
#define __H_ARGUMENT

#include "ArgumentType.h"
#include <string>

namespace io
{
	namespace readers
	{
		namespace input
		{
			namespace commandline
			{
				namespace arguments
				{
					struct Argument
					{
						std::string Name;
						ArgumentType Type;
						Argument(std::string Name, ArgumentType Type)
							: Name(Name), Type(Type)
						{

						}
					};
				}
			}
		}
	}
}

#endif