#ifndef SSPP_COMMON_MATRIXMARKETINFO_H_
#define SSPP_COMMON_MATRIXMARKETINFO_H_

#include <ostream>

struct MatrixMarketInfo {
  unsigned rows;
  unsigned columns;
  unsigned non_zeros;
  unsigned max_row_non_zeros;
  unsigned min_row_non_zeros;
  double average_row_non_zeros;
  double standard_deviation;

  MatrixMarketInfo(unsigned rows,
                   unsigned columns,
                   unsigned non_zeros,
                   unsigned max_row_non_zeros,
                   unsigned min_row_non_zeros,
                   double average_row_non_zeros,
                   double standard_deviation)
    : rows(rows),
    columns(columns),
    non_zeros(non_zeros),
    max_row_non_zeros(max_row_non_zeros),
    min_row_non_zeros(min_row_non_zeros),
    average_row_non_zeros(average_row_non_zeros),
    standard_deviation(standard_deviation) {
  }
  friend std::ostream & operator << (std::ostream & os, const MatrixMarketInfo & info) {
    os << "Rows: " << info.rows << '\n';
    os << "Columns: " << info.columns << '\n';
    os << "Non-zeros: " << info.non_zeros << '\n';
    os << "Max row non-zeros: " << info.max_row_non_zeros << '\n';
    os << "Min row non-zeros: " << info.min_row_non_zeros << '\n';
    os << "Average row non-zeros: " << info.average_row_non_zeros << '\n';
    os << "Standard deviation: " << info.standard_deviation << '\n';
    return os;
  }
};

#endif
