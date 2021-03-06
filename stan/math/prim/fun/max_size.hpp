#ifndef STAN_MATH_PRIM_FUN_MAX_SIZE_HPP
#define STAN_MATH_PRIM_FUN_MAX_SIZE_HPP

#include <stan/math/prim/fun/size.hpp>

namespace stan {
namespace math {

template <typename T1, typename T2>
size_t max_size(const T1& x1, const T2& x2) {
  size_t result = stan::math::size(x1);
  result = result > stan::math::size(x2) ? result : stan::math::size(x2);
  return result;
}

template <typename T1, typename T2, typename T3>
size_t max_size(const T1& x1, const T2& x2, const T3& x3) {
  size_t result = stan::math::size(x1);
  result = result > stan::math::size(x2) ? result : stan::math::size(x2);
  result = result > stan::math::size(x3) ? result : stan::math::size(x3);
  return result;
}

template <typename T1, typename T2, typename T3, typename T4>
size_t max_size(const T1& x1, const T2& x2, const T3& x3, const T4& x4) {
  size_t result = stan::math::size(x1);
  result = result > stan::math::size(x2) ? result : stan::math::size(x2);
  result = result > stan::math::size(x3) ? result : stan::math::size(x3);
  result = result > stan::math::size(x4) ? result : stan::math::size(x4);
  return result;
}

template <typename T1, typename T2, typename T3, typename T4, typename T5>
size_t max_size(const T1& x1, const T2& x2, const T3& x3, const T4& x4,
                const T5& x5) {
  size_t result = stan::math::size(x1);
  result = result > stan::math::size(x2) ? result : stan::math::size(x2);
  result = result > stan::math::size(x3) ? result : stan::math::size(x3);
  result = result > stan::math::size(x4) ? result : stan::math::size(x4);
  result = result > stan::math::size(x5) ? result : stan::math::size(x5);
  return result;
}

}  // namespace math
}  // namespace stan
#endif
