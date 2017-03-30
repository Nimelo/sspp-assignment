#ifndef SSPP_COMMON_MATRIXMARKETREADER_H_
#define SSPP_COMMON_MATRIXMARKETREADER_H_
#include "MatrixMarket.h"
#include "PatternValueResolverInterface.h"

namespace sspp {
  namespace common {
    class MatrixMarketReader {
    public:
      template<typename VALUE_TYPE>
      static MatrixMarket<VALUE_TYPE> Read(std::istream & stream, PatternValueResolverInterface<VALUE_TYPE> & resolver) {
        MatrixMarketHeader header;
        stream >> header;
        stream.clear();
        stream.seekg(0, stream.beg);
        while(stream.peek() == '%') stream.ignore(2048, '\n');
        unsigned rows, columns, non_zeros;
        stream >> rows >> columns >> non_zeros;
        std::vector<unsigned> row_indices, column_indices;
        std::vector<VALUE_TYPE> values;

        if(header.IsSymmetric()) {
          if(header.IsPattern()) {
            ReadSymmetricPattern(stream, resolver, non_zeros, row_indices, column_indices, values);
          } else {
            ReadSymmetric(stream, non_zeros, row_indices, column_indices, values);
          }
          non_zeros = values.size();
        } else {
          if(header.IsPattern()) {
            ReadGeneralPattern(stream, resolver, non_zeros, row_indices, column_indices, values);
          } else {
            ReadGeneral(stream, non_zeros, row_indices, column_indices, values);
          }
        }
        return MatrixMarket<VALUE_TYPE>(rows, columns, non_zeros, row_indices, column_indices, values);
      }
    };

    template<typename VALUE_TYPE>
    static void ReadSymmetricPattern(std::istream & stream, PatternValueResolverInterface<VALUE_TYPE> & resolver, unsigned non_zeros, std::vector<unsigned> & row_indices, std::vector<unsigned> & column_indices, std::vector<VALUE_TYPE> & values) {
      ReadGeneralPattern(stream, resolver, non_zeros, row_indices, column_indices, values);
      unsigned symmetric_non_zeros = 0;
      for(unsigned i = 0; i < non_zeros; i++) {
        if(row_indices[i] != column_indices[i]) {
          ++symmetric_non_zeros;
        }
      }
      row_indices.resize(non_zeros + symmetric_non_zeros);
      column_indices.resize(non_zeros + symmetric_non_zeros);
      values.resize(non_zeros + symmetric_non_zeros);

      unsigned index = 0;
      for(unsigned i = 0; i < non_zeros; i++) {
        if(row_indices[i] != column_indices[i]) {
          row_indices[index + non_zeros] = column_indices[i];
          column_indices[index + non_zeros] = row_indices[i];
          values[index + non_zeros] = values[i];
          ++index;
        }
      }
    }

    template<typename VALUE_TYPE>
    static void ReadSymmetric(std::istream & stream, unsigned non_zeros, std::vector<unsigned> & row_indices, std::vector<unsigned> & column_indices, std::vector<VALUE_TYPE> & values) {
      ReadGeneral(stream, non_zeros, row_indices, column_indices, values);
      unsigned symmetric_non_zeros = 0;
      for(unsigned i = 0; i < non_zeros; i++) {
        if(row_indices[i] != column_indices[i]) {
          ++symmetric_non_zeros;
        }
      }
      row_indices.resize(non_zeros + symmetric_non_zeros);
      column_indices.resize(non_zeros + symmetric_non_zeros);
      values.resize(non_zeros + symmetric_non_zeros);

      unsigned index = 0;
      for(unsigned i = 0; i < non_zeros; i++) {
        if(row_indices[i] != column_indices[i]) {
          row_indices[index + non_zeros] = column_indices[i];
          column_indices[index + non_zeros] = row_indices[i];
          values[index + non_zeros] = values[i];
          ++index;
        }
      }
    }

    template<typename VALUE_TYPE>
    static void ReadGeneralPattern(std::istream & stream, PatternValueResolverInterface<VALUE_TYPE> & resolver, unsigned non_zeros, std::vector<unsigned> & row_indices, std::vector<unsigned> & column_indices, std::vector<VALUE_TYPE> & values) {
      row_indices.resize(non_zeros);
      column_indices.resize(non_zeros);
      values.resize(non_zeros);
      for(unsigned i = 0; i < non_zeros; i++) {
        stream >> row_indices[i] >> column_indices[i];
        --row_indices[i];
        --column_indices[i];
      }
      for(unsigned i = 0; i < non_zeros; i++) {
        values[i] = resolver.GetPatternValue();
      }
    }

    template<typename VALUE_TYPE>
    static void ReadGeneral(std::istream & stream, unsigned non_zeros, std::vector<unsigned> & row_indices, std::vector<unsigned> & column_indices, std::vector<VALUE_TYPE> & values) {
      row_indices.resize(non_zeros);
      column_indices.resize(non_zeros);
      values.resize(non_zeros);
      for(unsigned i = 0; i < non_zeros; i++) {
        stream >> row_indices[i] >> column_indices[i] >> values[i];
        --row_indices[i];
        --column_indices[i];
      }
    }
  }
}
#endif
