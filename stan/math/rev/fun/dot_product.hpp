#ifndef STAN_MATH_REV_FUN_DOT_PRODUCT_HPP
#define STAN_MATH_REV_FUN_DOT_PRODUCT_HPP

#include <stan/math/rev/meta.hpp>
#include <stan/math/rev/core.hpp>
#include <stan/math/rev/fun/value_of.hpp>
#include <stan/math/prim/meta.hpp>
#include <stan/math/prim/err.hpp>
#include <stan/math/prim/fun/Eigen.hpp>
#include <stan/math/prim/fun/typedefs.hpp>
#include <stan/math/prim/fun/value_of.hpp>
#include <type_traits>
#include <vector>

namespace stan {
namespace math {
namespace internal {

template <typename T>
struct dot_product_store_type;

template <>
struct dot_product_store_type<var> {
  using type = vari**;
};

template <>
struct dot_product_store_type<double> {
  using type = double*;
};

template <typename T1, typename T2>
class dot_product_vari : public vari {
 protected:
  typename dot_product_store_type<T1>::type v1_;
  typename dot_product_store_type<T2>::type v2_;
  size_t length_;

  inline static double var_dot(vari** v1, vari** v2, size_t length) {
    Eigen::Map<vector_vi> vd1(v1, length);
    Eigen::Map<vector_vi> vd2(v2, length);

    return vd1.val().dot(vd2.val());
  }

  inline static double var_dot(const T1* v1, const T2* v2, size_t length) {
    Eigen::Map<const Eigen::Matrix<T1, -1, 1>> vd1(v1, length);
    Eigen::Map<const Eigen::Matrix<T2, -1, 1>> vd2(v2, length);
    return vd1.val().dot(vd2.val());
  }

  template <typename Derived1, typename Derived2>
  inline static double var_dot(const Eigen::DenseBase<Derived1>& v1,
                               const Eigen::DenseBase<Derived2>& v2) {
    vector_d vd1
        = Eigen::Ref<const Eigen::Matrix<typename Derived1::Scalar, -1, 1>>(v1)
              .val();
    vector_d vd2
        = Eigen::Ref<const Eigen::Matrix<typename Derived2::Scalar, -1, 1>>(v2)
              .val();
    return vd1.dot(vd2);
  }
  inline void chain(vari** v1, vari** v2) {
    Eigen::Map<vector_vi> vd1(v1, length_);
    Eigen::Map<vector_vi> vd2(v2, length_);
    vd1.adj() += adj_ * vd2.val();
    vd2.adj() += adj_ * vd1.val();
  }
  inline void chain(double* v1, vari** v2) {
    Eigen::Map<vector_vi>(v2, length_).adj()
        += adj_ * Eigen::Map<vector_d>(v1, length_);
  }
  inline void chain(vari** v1, double* v2) {
    Eigen::Map<vector_vi>(v1, length_).adj()
        += adj_ * Eigen::Map<vector_d>(v2, length_);
  }
  inline void initialize(vari**& mem_v, const var* inv,
                         vari** shared = nullptr) {
    if (shared == nullptr) {
      mem_v = reinterpret_cast<vari**>(
          ChainableStack::instance_->memalloc_.alloc(length_ * sizeof(vari*)));
      Eigen::Map<vector_vi>(mem_v, length_)
          = Eigen::Map<const vector_v>(inv, length_).vi();
    } else {
      mem_v = shared;
    }
  }
  template <typename Derived>
  inline void initialize(vari**& mem_v, const Eigen::DenseBase<Derived>& inv,
                         vari** shared = nullptr) {
    if (shared == nullptr) {
      mem_v = reinterpret_cast<vari**>(
          ChainableStack::instance_->memalloc_.alloc(length_ * sizeof(vari*)));
      Eigen::Map<vector_vi>(mem_v, length_)
          = Eigen::Ref<const vector_v>(inv).vi();
    } else {
      mem_v = shared;
    }
  }

  inline void initialize(double*& mem_d, const double* ind,
                         double* shared = nullptr) {
    if (shared == nullptr) {
      mem_d = reinterpret_cast<double*>(
          ChainableStack::instance_->memalloc_.alloc(length_ * sizeof(double)));
      for (size_t i = 0; i < length_; i++) {
        mem_d[i] = ind[i];
      }
    } else {
      mem_d = shared;
    }
  }
  template <typename Derived>
  inline void initialize(double*& mem_d, const Eigen::DenseBase<Derived>& ind,
                         double* shared = nullptr) {
    if (shared == nullptr) {
      mem_d = reinterpret_cast<double*>(
          ChainableStack::instance_->memalloc_.alloc(length_ * sizeof(double)));
      Eigen::Map<vector_d>(mem_d, length_) = Eigen::Ref<const vector_d>(ind);
    } else {
      mem_d = shared;
    }
  }

 public:
  dot_product_vari(typename dot_product_store_type<T1>::type v1,
                   typename dot_product_store_type<T2>::type v2, size_t length)
      : vari(var_dot(v1, v2, length)), v1_(v1), v2_(v2), length_(length) {}

  dot_product_vari(const T1* v1, const T2* v2, size_t length,
                   dot_product_vari<T1, T2>* shared_v1 = NULL,
                   dot_product_vari<T1, T2>* shared_v2 = NULL)
      : vari(var_dot(v1, v2, length)), length_(length) {
    if (shared_v1 == NULL) {
      initialize(v1_, v1);
    } else {
      initialize(v1_, v1, shared_v1->v1_);
    }
    if (shared_v2 == NULL) {
      initialize(v2_, v2);
    } else {
      initialize(v2_, v2, shared_v2->v2_);
    }
  }
  template <typename Derived1, typename Derived2>
  dot_product_vari(const Eigen::DenseBase<Derived1>& v1,
                   const Eigen::DenseBase<Derived2>& v2,
                   dot_product_vari<T1, T2>* shared_v1 = NULL,
                   dot_product_vari<T1, T2>* shared_v2 = NULL)
      : vari(var_dot(v1, v2)), length_(v1.size()) {
    if (shared_v1 == NULL) {
      initialize(v1_, v1);
    } else {
      initialize(v1_, v1, shared_v1->v1_);
    }
    if (shared_v2 == NULL) {
      initialize(v2_, v2);
    } else {
      initialize(v2_, v2, shared_v2->v2_);
    }
  }
  template <int R1, int C1, int R2, int C2>
  dot_product_vari(const Eigen::Matrix<T1, R1, C1>& v1,
                   const Eigen::Matrix<T2, R2, C2>& v2,
                   dot_product_vari<T1, T2>* shared_v1 = NULL,
                   dot_product_vari<T1, T2>* shared_v2 = NULL)
      : vari(var_dot(v1, v2)), length_(v1.size()) {
    if (shared_v1 == NULL) {
      initialize(v1_, v1);
    } else {
      initialize(v1_, v1, shared_v1->v1_);
    }
    if (shared_v2 == NULL) {
      initialize(v2_, v2);
    } else {
      initialize(v2_, v2, shared_v2->v2_);
    }
  }
  virtual void chain() { chain(v1_, v2_); }
};
}  // namespace internal

/**
 * Returns the dot product.
 *
 * @tparam T1 type of the first vector
 * @tparam T2 type of the second vector
 *
 * @param[in] v1 First row or column vector.
 * @param[in] v2 Second row or column vector.
 * @return Dot product of the vectors.
 * @throw std::domain_error if length of v1 is not equal to length of v2.
 */
template <typename Vec1, typename Vec2,
          typename = require_all_eigen_vector_t<Vec1, Vec2>,
          typename = require_any_eigen_vt<is_var, Vec1, Vec2>, typename = void>
inline auto dot_product(const Vec1& v1, const Vec2& v2) {
  check_matching_sizes("dot_product", "v1", v1, "v2", v2);
  return var(
      new internal::dot_product_vari<value_type_t<Vec1>, value_type_t<Vec2>>(
          v1, v2));
}

/**
 * Returns the dot product.
 *
 * @tparam T1 type of elements in the first vector
 * @tparam T2 type of elements in the second vector
 *
 * @param[in] v1 First array.
 * @param[in] v2 Second array.
 * @param[in] length Length of both arrays.
 * @return Dot product of the arrays.
 */
template <typename T1, typename T2, typename = require_any_var_t<T1, T2>>
inline return_type_t<T1, T2> dot_product(const T1* v1, const T2* v2,
                                         size_t length) {
  return var(new internal::dot_product_vari<T1, T2>(v1, v2, length));
}

/**
 * Returns the dot product.
 *
 * @tparam T1 type of elements in the first vector
 * @tparam T2 type of elements in the second vector
 *
 * @param[in] v1 First vector.
 * @param[in] v2 Second vector.
 * @return Dot product of the vectors.
 * @throw std::domain_error if sizes of v1 and v2 do not match.
 */
template <typename T1, typename T2, typename = require_any_var_t<T1, T2>>
inline return_type_t<T1, T2> dot_product(const std::vector<T1>& v1,
                                         const std::vector<T2>& v2) {
  check_matching_sizes("dot_product", "v1", v1, "v2", v2);
  return var(new internal::dot_product_vari<T1, T2>(&v1[0], &v2[0], v1.size()));
}

}  // namespace math
}  // namespace stan
#endif
