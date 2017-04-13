#pragma once

#include "SerialParallelComparison.h"
#include <ostream>
#include <map>
#include <vector>

class OpenMPCumulativeResult {
public:
  double crs_float_max = 0;
  double crs_double_max = 0;
  double ellpack_float_max = 0;
  double ellpack_double_max = 0;

  OpenMPCumulativeResult() {}
  OpenMPCumulativeResult(const OpenMPCumulativeResult & other) {
    thread_entries_ = other.thread_entries_;
    thread_results_crs_double_ = other.thread_results_crs_double_;
    thread_results_crs_float_ = other.thread_results_crs_float_;
    thread_results_ellpack_double_ = other.thread_results_ellpack_double_;
    thread_results_ellpack_float_ = other.thread_results_ellpack_float_;
    crs_float_max = other.crs_float_max;
    crs_double_max = other.crs_double_max;
    ellpack_float_max = other.ellpack_float_max;
    ellpack_double_max = other.ellpack_double_max;
  }

  void InsertResult(unsigned thread_number,
                    SerialParallelComparison crs_float,
                    SerialParallelComparison crs_double,
                    SerialParallelComparison ellpack_float,
                    SerialParallelComparison ellpack_double
  ) {
    thread_entries_.push_back(thread_number);
    thread_results_crs_float_.insert({ thread_number, crs_float });
    thread_results_crs_double_.insert({ thread_number, crs_double });
    thread_results_ellpack_float_.insert({ thread_number, ellpack_float });
    thread_results_ellpack_double_.insert({ thread_number, ellpack_double });
    crs_float_max = crs_float_max > crs_float.GetParallelOps() ? crs_float_max : crs_float.GetParallelOps();
    crs_double_max = crs_double_max > crs_double.GetParallelOps() ? crs_double_max : crs_double.GetParallelOps();
    ellpack_float_max = ellpack_float_max > ellpack_float.GetParallelOps() ? ellpack_float_max : ellpack_float.GetParallelOps();
    ellpack_double_max = ellpack_double_max > ellpack_double.GetParallelOps() ? ellpack_double_max : ellpack_double.GetParallelOps();
  }
  friend std::ostream & operator << (std::ostream & os, const OpenMPCumulativeResult & result) {

    os << "threads" << ' ' << "crs_float" << ' ' << "crs_double" << ' ' << "ellpack_float" << ' ' << "ellpack_double" << std::endl;
    for(auto thread_number : result.thread_entries_) {
      os << thread_number << ' ' << result.thread_results_crs_float_.find(thread_number)->second.GetParallelOps() << ' ';
      os << result.thread_results_crs_double_.find(thread_number)->second.GetParallelOps() << ' ';
      os << result.thread_results_ellpack_float_.find(thread_number)->second.GetParallelOps() << ' ';
      os << result.thread_results_ellpack_double_.find(thread_number)->second.GetParallelOps() << std::endl;
    }

    return os;
  }
protected:
  std::vector<unsigned> thread_entries_;
  std::map<unsigned, SerialParallelComparison> thread_results_crs_float_;
  std::map<unsigned, SerialParallelComparison> thread_results_ellpack_float_;
  std::map<unsigned, SerialParallelComparison> thread_results_crs_double_;
  std::map<unsigned, SerialParallelComparison> thread_results_ellpack_double_;
};
