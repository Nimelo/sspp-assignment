#pragma once
#include <string>
#include <map>
#include "CRS.h"
#include "MatrixMarket.h"
#include "MarketMatrixReader.h"
#include <fstream>
#include "DoublePatternResolver.h"
#include "CRSTransformer.h"
#include "ELLPACK.h"
#include "ELLPACKTransformer.h"

class MatrixContainer {
public:
  bool LoadOnce(std::string path, std::string key) {
    bool found = matrix_map_.find(key) != matrix_map_.end();
    if(!found) {
      std::ifstream is(path);
      sspp::common::DoublePatternResolver resolver;
      sspp::common::MatrixMarket<double> matrix_market = sspp::common::MatrixMarketReader::Read(is, resolver);
      matrix_map_.insert({ key, matrix_market });
      is.close();
    }
    return found;
  }

  bool Exist(std::string key) {
    return matrix_map_.find(key) != matrix_map_.end();
  }

  void Delete(std::string key) {
    matrix_map_.erase(key);
  }

  static MatrixContainer & GetInstance() {
    static MatrixContainer instance;
    return instance;
  }

  template<typename T>
  sspp::common::CRS<T> GetCRS(std::string key) {
    return sspp::common::CRSTransformer::transform<T, double>(matrix_map_.find(key)->second);
  }

  template<typename T>
  sspp::common::ELLPACK<T> GetELLPACK(std::string key) {
    return sspp::common::ELLPACKTransformer::transform<T, double>(matrix_map_.find(key)->second);
  }

protected:
  MatrixContainer() = default;
  std::map<std::string, sspp::common::MatrixMarket<double>> matrix_map_;
};
