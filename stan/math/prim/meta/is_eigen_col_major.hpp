#ifndef STAN_MATH_PRIM_META_IS_EIGEN_COL_MAJOR_HPP
#define STAN_MATH_PRIM_META_IS_EIGEN_COL_MAJOR_HPP

#include <stan/math/prim/fun/Eigen.hpp>
#include <stan/math/prim/meta/bool_constant.hpp>
#include <stan/math/prim/meta/is_eigen_matrix_base.hpp>
#include <stan/math/prim/meta/require_helpers.hpp>
#include <type_traits>

namespace stan {
/**
 * Checks whether type T that is derived from Eigen::MatrixBase is in
 * column-major order.
 * If true this will have a static member function named value with a type
 * of true, else value is false.
 * @tparam T Type derived from Eigen::MatrixBase to check if it is in column
 * major order
 * @tparam Enable used for SFINAE deduction.
 * @ingroup type_trait
 */
template <typename T, require_eigen_t<T>* = nullptr>
struct is_eigen_col_major : bool_constant<!std::decay_t<T>::IsRowMajor> {};

/*! \ingroup require_eigens_types */
/*! \defgroup eigen_sparse_base_types eigen_sparse_base  */
/*! \addtogroup eigen_sparse_base_types */
/*! @{ */

/*! \brief Require type satisfies @ref is_eigen_col_major */
/*! @tparam T the type to check */
template <typename T>
using require_eigen_col_major_t
    = require_t<is_eigen_col_major<std::decay_t<T>>>;
/*! @} */

}  // namespace stan

#endif