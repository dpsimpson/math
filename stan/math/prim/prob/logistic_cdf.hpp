#ifndef STAN_MATH_PRIM_PROB_LOGISTIC_CDF_HPP
#define STAN_MATH_PRIM_PROB_LOGISTIC_CDF_HPP

#include <stan/math/prim/meta.hpp>
#include <stan/math/prim/err.hpp>
#include <stan/math/prim/fun/constants.hpp>
#include <stan/math/prim/fun/exp.hpp>
#include <stan/math/prim/fun/max_size.hpp>
#include <stan/math/prim/fun/size.hpp>
#include <stan/math/prim/fun/size_zero.hpp>
#include <stan/math/prim/fun/value_of.hpp>
#include <stan/math/prim/prob/logistic_log.hpp>
#include <cmath>

namespace stan {
namespace math {

// Logistic(y|mu, sigma) [sigma > 0]
template <typename T_y, typename T_loc, typename T_scale>
return_type_t<T_y, T_loc, T_scale> logistic_cdf(const T_y& y, const T_loc& mu,
                                                const T_scale& sigma) {
  using T_partials_return = partials_return_t<T_y, T_loc, T_scale>;
  using std::exp;
  static const char* function = "logistic_cdf";
  check_not_nan(function, "Random variable", y);
  check_finite(function, "Location parameter", mu);
  check_positive_finite(function, "Scale parameter", sigma);
  check_consistent_sizes(function, "Random variable", y, "Location parameter",
                         mu, "Scale parameter", sigma);

  if (size_zero(y, mu, sigma)) {
    return 1.0;
  }

  T_partials_return P(1.0);
  operands_and_partials<T_y, T_loc, T_scale> ops_partials(y, mu, sigma);

  scalar_seq_view<T_y> y_vec(y);
  scalar_seq_view<T_loc> mu_vec(mu);
  scalar_seq_view<T_scale> sigma_vec(sigma);
  size_t N = max_size(y, mu, sigma);

  // Explicit return for extreme values
  // The gradients are technically ill-defined, but treated as zero
  for (size_t i = 0; i < stan::math::size(y); i++) {
    if (value_of(y_vec[i]) == NEGATIVE_INFTY) {
      return ops_partials.build(0.0);
    }
  }

  for (size_t n = 0; n < N; n++) {
    // Explicit results for extreme values
    // The gradients are technically ill-defined, but treated as zero
    if (value_of(y_vec[n]) == INFTY) {
      continue;
    }

    const T_partials_return y_dbl = value_of(y_vec[n]);
    const T_partials_return mu_dbl = value_of(mu_vec[n]);
    const T_partials_return sigma_dbl = value_of(sigma_vec[n]);
    const T_partials_return sigma_inv_vec = 1.0 / value_of(sigma_vec[n]);

    const T_partials_return Pn
        = 1.0 / (1.0 + exp(-(y_dbl - mu_dbl) * sigma_inv_vec));

    P *= Pn;

    if (!is_constant_all<T_y>::value) {
      ops_partials.edge1_.partials_[n]
          += exp(logistic_log(y_dbl, mu_dbl, sigma_dbl)) / Pn;
    }
    if (!is_constant_all<T_loc>::value) {
      ops_partials.edge2_.partials_[n]
          += -exp(logistic_log(y_dbl, mu_dbl, sigma_dbl)) / Pn;
    }
    if (!is_constant_all<T_scale>::value) {
      ops_partials.edge3_.partials_[n]
          += -(y_dbl - mu_dbl) * sigma_inv_vec
             * exp(logistic_log(y_dbl, mu_dbl, sigma_dbl)) / Pn;
    }
  }

  if (!is_constant_all<T_y>::value) {
    for (size_t n = 0; n < stan::math::size(y); ++n) {
      ops_partials.edge1_.partials_[n] *= P;
    }
  }
  if (!is_constant_all<T_loc>::value) {
    for (size_t n = 0; n < stan::math::size(mu); ++n) {
      ops_partials.edge2_.partials_[n] *= P;
    }
  }
  if (!is_constant_all<T_scale>::value) {
    for (size_t n = 0; n < stan::math::size(sigma); ++n) {
      ops_partials.edge3_.partials_[n] *= P;
    }
  }
  return ops_partials.build(P);
}

}  // namespace math
}  // namespace stan
#endif
