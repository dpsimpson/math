#ifndef STAN_MATH_PRIM_SPARSE_MATCH_PATTERN_HPP
#define STAN_MATH_PRIM_SPARSE_MATCH_PATTERN_HPP

#include <stan/math/prim/fun/Eigen.hpp>
#include <stan/math/prim/meta.hpp>

namespace stan {
namespace math {
namespace internal {

/*
RAII functor class that returns a sparse matrix with the values of
A and the non-zero structure of pattern.
NOTE: It is assumed that the non-zeros of pattern are a subset of
the non-zeros of A. (ie  pattern[i,j] != 0 => A[i,j] !=0)
*/

template <typename SpMat, require_eigen_sparse_base_t<SpMat>* = nullptr>
class MatchPattern {
  using T = typename SpMat::value_type;
  using MatrixType = typename stan::plain_type_t<SpMat>;
  using StorageIndex = typename MatrixType::StorageIndex;
  StorageIndex* m_outer;
  StorageIndex* m_inner;
  T* m_val;
  StorageIndex m_cols;
  StorageIndex m_nnz;

 public:
  MatchPattern(const SpMat& A, const SpMat& pattern) {
    m_cols = pattern.cols();
    m_nnz = pattern.nonZeros();

    m_outer = new StorageIndex[m_cols + 1];
    std::copy(pattern.outerIndexPtr(), pattern.outerIndexPtr() + m_cols + 1,
              m_outer);
    m_inner = new StorageIndex[m_nnz];
    std::copy(pattern.innerIndexPtr(), pattern.innerIndexPtr() + m_nnz,
              m_inner);
    m_val = new T[m_nnz];

    T* valptr = m_val;
    for (int j = 0; j < m_cols; ++j) {
      typename SpMat::InnerIterator Acol(A, j);
      for (typename SpMat::InnerIterator pattern_col(pattern, j);
           pattern_col; ++pattern_col) {
        while (Acol && (Acol.row() < pattern_col.row())) {
          ++Acol;
        }
        *valptr++ = Acol.value();
        ++Acol;
      }
    }
  }

  // Specialization for rank-1 matrces A = bc^T
  MatchPattern(const typename Eigen::Matrix<T, 1, Eigen::Dynamic>& b,
               const typename Eigen::Matrix<T, 1, Eigen::Dynamic>& c,
               const SpMat& pattern) {
    m_cols = pattern.cols();
    m_nnz = pattern.nonZeros();
    m_outer = new StorageIndex[m_cols + 1];
    std::copy(pattern.outerIndexPtr(), pattern.outerIndexPtr() + m_cols + 1,
              m_outer);
    m_inner = new StorageIndex[m_nnz];
    std::copy(pattern.innerIndexPtr(), pattern.innerIndexPtr() + m_nnz,
              m_inner);
    m_val = new T[m_nnz];

    T* valptr = m_val;
    for (int j = 0; j < m_cols; ++j) {
      for (typename SpMat::InnerIterator pattern_col(pattern, j); pattern_col;
           ++pattern_col) {
        *valptr++ = b.coeff(pattern_col.row()) * c.coeff(j);
      }
    }
  }

  ~MatchPattern() {
    delete[] m_inner;
    delete[] m_outer;
    delete[] m_val;
  }

  SpMat operator()() {
    return Eigen::Map<SpMat>(m_cols, m_cols, m_nnz, m_outer, m_inner, m_val);
  }
};

}  // namespace internal
}  // namespace math
}  // namespace stan

#endif