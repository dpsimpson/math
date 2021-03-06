#ifndef STAN_MATH_PRIM_PROB_BETA_LPDF_HPP
#define STAN_MATH_PRIM_PROB_BETA_LPDF_HPP

#include <stan/math/prim/meta.hpp>
#include <stan/math/prim/err.hpp>
#include <stan/math/prim/fun/constants.hpp>
#include <stan/math/prim/fun/digamma.hpp>
#include <stan/math/prim/fun/lgamma.hpp>
#include <stan/math/prim/fun/log.hpp>
#include <stan/math/prim/fun/log1m.hpp>
#include <stan/math/prim/fun/max_size.hpp>
#include <stan/math/prim/fun/multiply_log.hpp>
#include <stan/math/prim/fun/size.hpp>
#include <stan/math/prim/fun/size_zero.hpp>
#include <stan/math/prim/fun/value_of.hpp>
#include <cmath>

namespace stan {
namespace math {

/** \ingroup prob_dists
 * The log of the beta density for the specified scalar(s) given the specified
 * sample stan::math::size(s). y, alpha, or beta can each either be scalar or a
 * vector. Any vector inputs must be the same length.
 *
 * <p> The result log probability is defined to be the sum of
 * the log probabilities for each observation/alpha/beta triple.
 *
 * Prior sample sizes, alpha and beta, must be greater than 0.
 *
 * @tparam T_y type of scalar outcome
 * @tparam T_scale_succ type of prior scale for successes
 * @tparam T_scale_fail type of prior scale for failures
 * @param y (Sequence of) scalar(s).
 * @param alpha (Sequence of) prior sample stan::math::size(s).
 * @param beta (Sequence of) prior sample stan::math::size(s).
 * @return The log of the product of densities.
 */
template <bool propto, typename T_y, typename T_scale_succ,
          typename T_scale_fail>
return_type_t<T_y, T_scale_succ, T_scale_fail> beta_lpdf(
    const T_y& y, const T_scale_succ& alpha, const T_scale_fail& beta) {
  using T_partials_return = partials_return_t<T_y, T_scale_succ, T_scale_fail>;
  using std::log;
  static const char* function = "beta_lpdf";
  check_positive_finite(function, "First shape parameter", alpha);
  check_positive_finite(function, "Second shape parameter", beta);
  check_not_nan(function, "Random variable", y);
  check_consistent_sizes(function, "Random variable", y,
                         "First shape parameter", alpha,
                         "Second shape parameter", beta);
  check_nonnegative(function, "Random variable", y);
  check_less_or_equal(function, "Random variable", y, 1);

  if (size_zero(y, alpha, beta)) {
    return 0;
  }
  if (!include_summand<propto, T_y, T_scale_succ, T_scale_fail>::value) {
    return 0;
  }

  T_partials_return logp(0);
  operands_and_partials<T_y, T_scale_succ, T_scale_fail> ops_partials(y, alpha,
                                                                      beta);
  scalar_seq_view<T_y> y_vec(y);
  scalar_seq_view<T_scale_succ> alpha_vec(alpha);
  scalar_seq_view<T_scale_fail> beta_vec(beta);
  size_t size_y = stan::math::size(y);
  size_t size_alpha = stan::math::size(alpha);
  size_t size_beta = stan::math::size(beta);
  size_t N = max_size(y, alpha, beta);

  for (size_t n = 0; n < N; n++) {
    const T_partials_return y_dbl = value_of(y_vec[n]);
    if (y_dbl < 0 || y_dbl > 1) {
      return LOG_ZERO;
    }
  }

  VectorBuilder<include_summand<propto, T_y, T_scale_succ>::value,
                T_partials_return, T_y>
      log_y(size_y);
  VectorBuilder<include_summand<propto, T_y, T_scale_fail>::value,
                T_partials_return, T_y>
      log1m_y(size_y);

  for (size_t n = 0; n < size_y; n++) {
    if (include_summand<propto, T_y, T_scale_succ>::value) {
      log_y[n] = log(value_of(y_vec[n]));
    }
    if (include_summand<propto, T_y, T_scale_fail>::value) {
      log1m_y[n] = log1m(value_of(y_vec[n]));
    }
  }

  VectorBuilder<include_summand<propto, T_scale_succ>::value, T_partials_return,
                T_scale_succ>
      lgamma_alpha(size_alpha);
  VectorBuilder<!is_constant_all<T_scale_succ>::value, T_partials_return,
                T_scale_succ>
      digamma_alpha(size_alpha);
  if (include_summand<propto, T_scale_succ>::value) {
    for (size_t n = 0; n < size_alpha; n++) {
      const T_partials_return alpha_dbl = value_of(alpha_vec[n]);
      lgamma_alpha[n] = lgamma(alpha_dbl);
      if (!is_constant_all<T_scale_succ>::value) {
        digamma_alpha[n] = digamma(alpha_dbl);
      }
    }
  }

  VectorBuilder<include_summand<propto, T_scale_fail>::value, T_partials_return,
                T_scale_fail>
      lgamma_beta(size_beta);
  VectorBuilder<!is_constant_all<T_scale_fail>::value, T_partials_return,
                T_scale_fail>
      digamma_beta(size_beta);

  if (include_summand<propto, T_scale_fail>::value) {
    for (size_t n = 0; n < size_beta; n++) {
      const T_partials_return beta_dbl = value_of(beta_vec[n]);
      lgamma_beta[n] = lgamma(beta_dbl);
      if (!is_constant_all<T_scale_fail>::value) {
        digamma_beta[n] = digamma(beta_dbl);
      }
    }
  }

  VectorBuilder<include_summand<propto, T_scale_succ, T_scale_fail>::value,
                T_partials_return, T_scale_succ, T_scale_fail>
      lgamma_alpha_beta(max_size(alpha, beta));
  VectorBuilder<!is_constant_all<T_scale_succ, T_scale_fail>::value,
                T_partials_return, T_scale_succ, T_scale_fail>
      digamma_alpha_beta(max_size(alpha, beta));

  if (include_summand<propto, T_scale_succ, T_scale_fail>::value) {
    for (size_t n = 0; n < max_size(alpha, beta); n++) {
      const T_partials_return alpha_beta
          = value_of(alpha_vec[n]) + value_of(beta_vec[n]);
      lgamma_alpha_beta[n] = lgamma(alpha_beta);
      if (!is_constant_all<T_scale_succ, T_scale_fail>::value) {
        digamma_alpha_beta[n] = digamma(alpha_beta);
      }
    }
  }

  for (size_t n = 0; n < N; n++) {
    const T_partials_return y_dbl = value_of(y_vec[n]);
    const T_partials_return alpha_dbl = value_of(alpha_vec[n]);
    const T_partials_return beta_dbl = value_of(beta_vec[n]);

    if (include_summand<propto, T_scale_succ, T_scale_fail>::value) {
      logp += lgamma_alpha_beta[n];
    }
    if (include_summand<propto, T_scale_succ>::value) {
      logp -= lgamma_alpha[n];
    }
    if (include_summand<propto, T_scale_fail>::value) {
      logp -= lgamma_beta[n];
    }
    if (include_summand<propto, T_y, T_scale_succ>::value) {
      logp += (alpha_dbl - 1.0) * log_y[n];
    }
    if (include_summand<propto, T_y, T_scale_fail>::value) {
      logp += (beta_dbl - 1.0) * log1m_y[n];
    }

    if (!is_constant_all<T_y>::value) {
      ops_partials.edge1_.partials_[n]
          += (alpha_dbl - 1) / y_dbl + (beta_dbl - 1) / (y_dbl - 1);
    }
    if (!is_constant_all<T_scale_succ>::value) {
      ops_partials.edge2_.partials_[n]
          += log_y[n] + digamma_alpha_beta[n] - digamma_alpha[n];
    }
    if (!is_constant_all<T_scale_fail>::value) {
      ops_partials.edge3_.partials_[n]
          += log1m_y[n] + digamma_alpha_beta[n] - digamma_beta[n];
    }
  }
  return ops_partials.build(logp);
}

template <typename T_y, typename T_scale_succ, typename T_scale_fail>
inline return_type_t<T_y, T_scale_succ, T_scale_fail> beta_lpdf(
    const T_y& y, const T_scale_succ& alpha, const T_scale_fail& beta) {
  return beta_lpdf<false>(y, alpha, beta);
}

}  // namespace math
}  // namespace stan
#endif
