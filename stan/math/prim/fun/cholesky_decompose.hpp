#ifndef STAN_MATH_PRIM_FUN_CHOLESKY_DECOMPOSE_HPP
#define STAN_MATH_PRIM_FUN_CHOLESKY_DECOMPOSE_HPP

#include <stan/math/prim/meta.hpp>
#include <stan/math/prim/err.hpp>
#include <stan/math/prim/fun/Eigen.hpp>
#include <stan/math/prim/fun/to_ref.hpp>

#include <cmath>

namespace stan {
namespace math {

/**
 * Return the lower-triangular Cholesky factor (i.e., matrix
 * square root) of the specified square, symmetric matrix.  The return
 * value \f$L\f$ will be a lower-triangular matrix such that the
 * original matrix \f$A\f$ is given by
 * <p>\f$A = L \times L^EigMat\f$.
 *
 * @tparam EigMat type of the matrix (must be derived from \c Eigen::MatrixBase)
 * @param m Symmetric matrix.
 * @return Square root of matrix.
 * @note Because OpenCL only works on doubles there are two
 * <code>cholesky_decompose</code> functions. One that works on doubles
 * and another that works on all other types (this one).
 * @throw std::domain_error if m is not a symmetric matrix or
 *   if m is not positive definite (if m has more than 0 elements)
 */
template <typename EigMat, require_eigen_matrix_base_t<EigMat>* = nullptr,
          require_not_eigen_vt<is_var, EigMat>* = nullptr>
inline Eigen::Matrix<value_type_t<EigMat>, EigMat::RowsAtCompileTime,
                     EigMat::ColsAtCompileTime>
cholesky_decompose(const EigMat& m) {
  const eval_return_type_t<EigMat>& m_eval = m.eval();
  check_symmetric("cholesky_decompose", "m", m_eval);
  check_not_nan("cholesky_decompose", "m", m_eval);
  Eigen::LLT<Eigen::Matrix<value_type_t<EigMat>, EigMat::RowsAtCompileTime,
                           EigMat::ColsAtCompileTime>>
      llt = m_eval.llt();
  check_pos_definite("cholesky_decompose", "m", llt);
  return llt.matrixL();
}

/**
 * Return the sparse lower-triangular Cholesky factor (i.e., matrix
 * square root) of the specified sparse square, symmetric matrix.  The return
 * value is a tuple \f$(L, P)\f$, where \f$L\f$  a sparse lower-triangular
 * matrix and \f$P\f$ is a permutation of \f$[1,\ldots,n]\f$ such that the
 * original matrix \f$A\f$ is given by
 * <p>\f$A = P^TL  L^TP\f$,
 * where \f$P\f$ is the permutation matrix that has been computed to minimize
 * fill-in. The matrix \f$P\f$ is represented as a `std::vector<int>`.
 *
 * @tparam SpEigMat type of the matrix (must be derived from \c
 * Eigen::SparseMatrixBase)
 * @param m Sparse matrix. The matrix is assumed to be symmetric with its
 * non-zero elements stored in its lower triangle. Elements in the upper
 * triangle of m will not be read.
 * @return A tuple containing the Cholesky triangle of \f$PAP^T\f$ and the
 * permutation \f$P\f$ represented as a `std::vector<int>`.
 * @throw std::domain_error if m is not a symmetric matrix or
 *   if m is not positive definite (if m has more than 0 elements)
 */
template <typename SpMat, require_eigen_sparse_base_t<SpMat>* = nullptr,
          require_eigen_col_major_t<SpMat>* = nullptr,
          require_not_eigen_vt<is_var, SpMat>* = nullptr>
inline std::tuple<plain_type_t<SpMat>, std::vector<int>> cholesky_decompose(
    const SpMat& m) {
  using SpMatOut = plain_type_t<SpMat>;

  const auto& m_eval = to_ref(m);

  check_square("cholesky_decompose", "m", m_eval);
  check_not_nan("cholesky_decompose", "m", m_eval);

  Eigen::SimplicialLLT<SpMatOut> llt(m_eval);
  check_pos_definite("cholesky_decompose", "m", llt);

  std::vector<int> perm(llt.permutationP().indices().data(),
                        llt.permutationP().indices().data()
                            + llt.permutationP().indices().size());

  return std::tuple<SpMatOut, std::vector<int>>(llt.matrixL(), perm);
}

}  // namespace math
}  // namespace stan

#endif
