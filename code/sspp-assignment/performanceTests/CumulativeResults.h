#pragma once
#include <ostream>
#include <string>
#include <map>

struct CudaOpenMPOPS {
  double crs_float_openmp_;
  double crs_double_openmp_;
  double crs_float_cuda_;
  double crs_double_cuda_;
  double ellpack_float_openmp_;
  double ellpack_double_openmp_;
  double ellpack_float_cuda_;
  double ellpack_double_cuda_;

  friend std::ostream & operator << (std::ostream & os, const CudaOpenMPOPS & result) {
    os << result.crs_float_openmp_ << ' ';
    os << result.crs_float_cuda_ << ' ';
    os << result.ellpack_float_openmp_ << ' ';
    os << result.ellpack_float_cuda_ << ' ';

    os << result.crs_double_openmp_ << ' ';
    os << result.crs_double_cuda_ << ' ';
    os << result.ellpack_double_openmp_ << ' ';
    os << result.ellpack_double_cuda_;
    return os;
  }
};

class CumulativeResults {
public:
  static CumulativeResults & GetInstance() {
    static CumulativeResults instance;
    return instance;
  }

  void AddCuda(std::string key,
               double crs_float,
               double crs_double,
               double ellpack_float,
               double ellpack_double) {
    if(cumulative_map_.find(key) == cumulative_map_.end()) {
      cumulative_map_.insert({ key, CudaOpenMPOPS() });
    }

    auto &entry = cumulative_map_.find(key)->second;
    entry.crs_float_cuda_ = crs_float;
    entry.crs_double_cuda_ = crs_double;
    entry.ellpack_float_cuda_ = ellpack_float;
    entry.ellpack_double_cuda_ = ellpack_double;
  }

  void AddOpenMp(std::string key,
                 double crs_float,
                 double crs_double,
                 double ellpack_float,
                 double ellpack_double) {
    if(cumulative_map_.find(key) == cumulative_map_.end()) {
      cumulative_map_.insert({ key, CudaOpenMPOPS() });
    }

    auto &entry = cumulative_map_.find(key)->second;
    entry.crs_float_openmp_ = crs_float;
    entry.crs_double_openmp_ = crs_double;
    entry.ellpack_float_openmp_ = ellpack_float;
    entry.ellpack_double_openmp_ = ellpack_double;
  }

  friend std::ostream & operator <<(std::ostream & os, const CumulativeResults & result) {
    os << "matrix" << ' ' << "crs_float_openmp" << ' ' << "crs_float_cuda" << ' ' << "ellpack_float_openmp" << ' ' << "ellpack_float_cuda" << ' ';
    os << "crs_double_openmp" << ' ' << "crs_double_cuda" << ' ' << "ellpack_double_openmp" << ' ' << "ellpack_double_cuda" << std::endl;
    for(auto const& entry : result.cumulative_map_) {
      os << entry.first << ' ' << entry.second << std::endl;
    }
    return os;
  }
protected:
  CumulativeResults() = default;
  std::map<std::string, CudaOpenMPOPS> cumulative_map_;
};
