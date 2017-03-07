#include "CommandlineTest.h"
#include "Argument.h"
#include <gtest\gtest.h>

#define OUT_DIR_TAG "-outdir"
#define IN_DIR_TAG "-indir"
#define M_TAG "-m"
#define N_TAG "-n"
#define ITERATIONS_TAG "-iterations"
#define FILES_TAG "-files"

TEST_F(CommandLineParameterReaderTest, shouldParseStandardInput)
{
	using namespace io::readers::input::commandline;
	std::vector<arguments::Argument> arguments;

	arguments.push_back(arguments::Argument(OUT_DIR_TAG, arguments::ArgumentType::Single));
	arguments.push_back(arguments::Argument(IN_DIR_TAG, arguments::ArgumentType::Single));
	arguments.push_back(arguments::Argument(M_TAG, arguments::ArgumentType::Single));
	arguments.push_back(arguments::Argument(N_TAG, arguments::ArgumentType::Single));
	arguments.push_back(arguments::Argument(ITERATIONS_TAG, arguments::ArgumentType::Single));
	arguments.push_back(arguments::Argument(FILES_TAG, arguments::ArgumentType::Multiple));

	CommandLineParameterReader reader(arguments);
	
	const int argc = 14;
	const char *argv[argc] =
	{
		OUT_DIR_TAG, "Path",
		IN_DIR_TAG, "Path",
		M_TAG, "14",
		N_TAG, "12",
		ITERATIONS_TAG, "15",
		FILES_TAG, "file1", "file2", "file3"
	};

	reader.load(argc, argv);

	for (int i = 0; i < arguments.size(); i++)
	{
		ASSERT_EQ(true, reader.hasArgument(arguments[i].Name));
	}

	std::string outDirTag = reader.get(std::string(OUT_DIR_TAG));
	std::string inDirTag = reader.get(std::string(IN_DIR_TAG));
	int mTag = reader.get(std::string(M_TAG));
	int nTag = reader.get(std::string(N_TAG));
	int iTag = reader.get(std::string(ITERATIONS_TAG));
	std::vector<std::string> files = reader.get(std::string(FILES_TAG));

	ASSERT_EQ("Path", outDirTag);
	ASSERT_EQ("Path", inDirTag);
	ASSERT_EQ(14, mTag);
	ASSERT_EQ(12, nTag);
	ASSERT_EQ(15, iTag);
	
	for (int i = 0; i < 3; i++)
	{
		ASSERT_EQ(std::string("file" + std::to_string(i + 1)), files.at(i));
	}

}

TEST_F(CommandLineParameterReaderTest, shouldParseStandardInput2)
{
	using namespace io::readers::input::commandline;
	std::vector<arguments::Argument> arguments;

	arguments.push_back(arguments::Argument(OUT_DIR_TAG, arguments::ArgumentType::Single));
	arguments.push_back(arguments::Argument(IN_DIR_TAG, arguments::ArgumentType::Single));
	arguments.push_back(arguments::Argument(M_TAG, arguments::ArgumentType::Single));
	arguments.push_back(arguments::Argument(N_TAG, arguments::ArgumentType::Single));
	arguments.push_back(arguments::Argument(ITERATIONS_TAG, arguments::ArgumentType::Single));
	arguments.push_back(arguments::Argument(FILES_TAG, arguments::ArgumentType::Multiple));

	CommandLineParameterReader reader(arguments);

	const int argc = 14;
	const char *argv[argc] =
	{
		OUT_DIR_TAG, "Path",
		IN_DIR_TAG, "Path",
		M_TAG, "14",
		N_TAG, "12",
		FILES_TAG, "file1", "file2", "file3",
		ITERATIONS_TAG, "15"		
	};

	reader.load(argc, argv);

	for (int i = 0; i < arguments.size(); i++)
	{
		ASSERT_EQ(true, reader.hasArgument(arguments[i].Name));
	}

	std::string outDirTag = reader.get(std::string(OUT_DIR_TAG));
	std::string inDirTag = reader.get(std::string(IN_DIR_TAG));
	int mTag = reader.get(std::string(M_TAG));
	int nTag = reader.get(std::string(N_TAG));
	int iTag = reader.get(std::string(ITERATIONS_TAG));
	std::vector<std::string> files = reader.get(std::string(FILES_TAG));

	ASSERT_EQ("Path", outDirTag);
	ASSERT_EQ("Path", inDirTag);
	ASSERT_EQ(14, mTag);
	ASSERT_EQ(12, nTag);
	ASSERT_EQ(15, iTag);

	for (int i = 0; i < 3; i++)
	{
		ASSERT_EQ(std::string("file" + std::to_string(i + 1)), files.at(i));
	}
}