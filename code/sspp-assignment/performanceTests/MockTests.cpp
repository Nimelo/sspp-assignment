#include "MockTest.h"

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "Argument.h"
#include "CommandLineParameterReader.h"
#include "CRS.h"
#include "CRSOpenMPSolver.h"
#include "CRSCudaSolver.h"

#define OUT_DIR_TAG "-outdir"
#define IN_DIR_TAG "-indir"
#define M_TAG "-m"
#define N_TAG "-n"
#define ITERATIONS_TAG "-iterations"
#define FILES_TAG "-files"

TEST_F(MockTest, SERIAL) {
  using namespace sspp::common;
  std::vector<Argument> arguments;

  arguments.push_back(Argument(OUT_DIR_TAG, ArgumentType::Single));
  arguments.push_back(Argument(IN_DIR_TAG, ArgumentType::Single));
  arguments.push_back(Argument(M_TAG, ArgumentType::Single));
  arguments.push_back(Argument(N_TAG, ArgumentType::Single));
  arguments.push_back(Argument(ITERATIONS_TAG, ArgumentType::Single));
  arguments.push_back(Argument(FILES_TAG, ArgumentType::Multiple));

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

  reader.Load(argc, argv);

  for(int i = 0; i < arguments.size(); i++) {
    ASSERT_EQ(true, reader.HasArgument(arguments[i].name));
  }

  std::string outDirTag = reader.GetParameter(std::string(OUT_DIR_TAG));
  std::string inDirTag = reader.GetParameter(std::string(IN_DIR_TAG));
  int mTag = reader.GetParameter(std::string(M_TAG));
  int nTag = reader.GetParameter(std::string(N_TAG));
  int iTag = reader.GetParameter(std::string(ITERATIONS_TAG));
  std::vector<std::string> files = reader.GetParameter(std::string(FILES_TAG));

  ASSERT_EQ("Path", outDirTag);
  ASSERT_EQ("Path", inDirTag);
  ASSERT_EQ(14, mTag);
  ASSERT_EQ(12, nTag);
  ASSERT_EQ(15, iTag);

  for(int i = 0; i < 3; i++) {
    ASSERT_EQ(std::string("file" + std::to_string(i + 1)), files.at(i));
  }

}


TEST_F(MockTest, OPENMP) {
  const unsigned M = 4, N = 4, NZ = 7, THREADS = 2;
  std::vector<unsigned> IRP = { 0, 2, 4, 5, 7 },
    JA = { 0, 1, 1, 2, 2, 2, 3 };
  std::vector<float> AS = { 11, 12, 22, 23, 33, 43, 44 };
  sspp::common::CRS<float> csr(M, N, NZ, IRP, JA, AS);
  std::vector<float> B = { 1, 1, 1, 1 };
  std::vector<unsigned> correctX = { 23, 45, 33, 87 };
  sspp::openmp::CRSOpenMPSolver<float> csrParallelSolver;

  csrParallelSolver.SetThreads(THREADS);
  auto output = csrParallelSolver.Solve(csr, B);

  ASSERT_EQ(M, output.GetValues().size());
  ASSERT_THAT(output.GetValues(), ::testing::ElementsAre(23, 45, 33, 87));

}

TEST_F(MockTest, CUDA) {
  const unsigned M = 4, N = 4, NZ = 7, THREADS = 2;
  std::vector<unsigned> IRP = { 0, 2, 4, 5, 7 },
    JA = { 0, 1, 1, 2, 2, 2, 3 };
  std::vector<float> AS = { 11, 12, 22, 23, 33, 43, 44 };
  sspp::common::CRS<float> csr(M, N, NZ, IRP, JA, AS);
  std::vector<float> B = { 1, 1, 1, 1 };
  std::vector<float> correctX = { 23, 45, 33, 87 };
  sspp::tools::solvers::CRSCudaSolver csrParallelSolver;

  auto output = csrParallelSolver.Solve(csr, B);

  ASSERT_EQ(M, output.GetValues().size()) << "Size of output_ is incorrect";
  ASSERT_THAT(output.GetValues(), ::testing::ElementsAre(23, 45, 33, 87));
}