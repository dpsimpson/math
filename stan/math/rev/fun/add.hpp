#ifndef STAN_MATH_REV_FUN_MATRIX_ADD_HPP
#define STAN_MATH_REV_FUN_MATRIX_ADD_HPP

#include <stan/math/rev/meta.hpp>
#include <stan/math/rev/core.hpp>
#include <stan/math/prim.hpp>
#include <type_traits>
#include <iostream>

namespace stan {
namespace math {

/**
 * This is a subclass of the vari class for matrix
 * addition A + B, where A and B are N by M matrices.
 *
 * The class stores the structure of each matrix,
 * the double values of A and B, and pointers to
 * the varis for A and B if A or B is a var. It
 * also instatiates and stores pointers to varis
 * for all elements of A + B.
 *
 * @tparam Ta type of elements in matrix A
 * @tparam Tb type of elements in matrix B
 * @tparam R number of rows (common to A and B)
 * @tparam C number of columns (common to A and B)
 */

template <typename Ta, typename Tb, int R, int C>
class add_mat_vari : public vari {
 public:
  int rows_;
  int cols_;
  int size_;
  double* Ad_;
  double* Bd_;
  vari** variRefA_;
  vari** variRefB_;
  vari** variRefAplusB_;

  /**
   * Constructor for add_mat_vari.
   *
   * All memeory is allocated in ChainableStack's
   * stack_alloc arena.
   *
   * It is critical for the efficiency of this object
   * that the constructor create new varis that aren't
   * popped onto the var_stack_, but rather are
   * popped onto the var_nochain_stack_. This is
   * controlled by the second argument to
   * vari's constructor.
   *
   * @param A matrix
   * @param  B matrix
   **/
  add_mat_vari(const Eigen::Matrix<Ta, R, C>& A,
               const Eigen::Matrix<Tb, R, C>& B)
      : vari(0.0),
        rows_(A.rows()),
        cols_(B.cols()),
        size_(A.size()),
        Ad_(ChainableStack::instance_->memalloc_.alloc_array<double>(size_)),
        Bd_(ChainableStack::instance_->memalloc_.alloc_array<double>(size_)),
        variRefA_(
            ChainableStack::instance_->memalloc_.alloc_array<vari*>(size_)),
        variRefB_(
            ChainableStack::instance_->memalloc_.alloc_array<vari*>(size_)),
        variRefAplusB_(
            ChainableStack::instance_->memalloc_.alloc_array<vari*>(size_)) {
    using Eigen::Map;
    Map<matrix_vi>(variRefA_, rows_, cols_) = A.vi();
    Map<matrix_vi>(variRefB_, rows_, cols_) = B.vi();
    Map<matrix_d> Ad(Ad_, rows_, cols_);
    Map<matrix_d> Bd(Bd_, rows_, cols_);
    Ad = A.val();
    Bd = B.val();
    Map<matrix_vi>(variRefAplusB_, rows_, cols_)
        = (Ad + Bd).unaryExpr([](double x) { return new vari(x, false); });
  }

  virtual void chain() {
    using Eigen::Map;
    matrix_d adjAplusB(rows_, cols_);
    adjAplusB = Map<matrix_vi>(variRefAplusB_, rows_, cols_).adj();
    Map<matrix_vi>(variRefA_, rows_, cols_).adj() += adjAplusB;
    Map<matrix_vi>(variRefB_, rows_, cols_).adj() += adjAplusB;
  }
};

/**
 * This is a subclass of the vari class for matrix
 * addition A + B, where A is an N by M matrix of
 * doubles and and B is an N by M matrix of vars.
 *
 * The class stores the structure of each matrix,
 * the double values of A and B, and pointers to
 * the varis for A and B if A or B is a var. It
 * also instatiates and stores pointers to varis
 * for all elements of A + B.
 *
 * @tparam Tb type of elements in matrix B
 * @tparam R number of rows (common to A and B)
 * @tparam C number of columns (common to A and B)
 */

template <typename Tb, int R, int C>
class add_mat_vari<double, Tb, R, C> : public vari {
 public:
  int rows_;
  int cols_;
  int size_;
  double* Ad_;
  double* Bd_;
  vari** variRefB_;
  vari** variRefAplusB_;

  /**
   * Constructor for add_mat_vari.
   *
   * All memeory is allocated in ChainableStack's
   * stack_alloc arena.
   *
   * It is critical for the efficiency of this object
   * that the constructor create new varis that aren't
   * popped onto the var_stack_, but rather are
   * popped onto the var_nochain_stack_. This is
   * controlled by the second argument to
   * vari's constructor.
   *
   * @param A matrix
   * @param  B matrix
   **/
  add_mat_vari(const Eigen::Matrix<double, R, C>& A,
               const Eigen::Matrix<Tb, R, C>& B)
      : vari(0.0),
        rows_(A.rows()),
        cols_(B.cols()),
        size_(A.size()),
        Ad_(ChainableStack::instance_->memalloc_.alloc_array<double>(size_)),
        Bd_(ChainableStack::instance_->memalloc_.alloc_array<double>(size_)),
        variRefB_(
            ChainableStack::instance_->memalloc_.alloc_array<vari*>(size_)),
        variRefAplusB_(
            ChainableStack::instance_->memalloc_.alloc_array<vari*>(size_)) {
    using Eigen::Map;
    Map<matrix_vi>(variRefB_, rows_, cols_) = B.vi();
    Map<matrix_d> Ad(Ad_, rows_, cols_);
    Map<matrix_d> Bd(Bd_, rows_, cols_);
    Ad = A;
    Bd = B.val();
    Map<matrix_vi>(variRefAplusB_, rows_, cols_)
        = (Ad + Bd).unaryExpr([](double x) { return new vari(x, false); });
  }

  virtual void chain() {
    using Eigen::Map;
    matrix_d adjAplusB(rows_, cols_);
    adjAplusB = Map<matrix_vi>(variRefAplusB_, rows_, cols_).adj();
    Map<matrix_vi>(variRefB_, rows_, cols_).adj() += adjAplusB;
  }
};

/**
 * This is a subclass of the vari class for matrix
 * addition A + B, where A is an N by M matrix of vars
 * and B is an N by M matrix of doubles.
 *
 * The class stores the structure of each matrix,
 * the double values of A and B, and pointers to
 * the varis for A and B if A or B is a var. It
 * also instatiates and stores pointers to varis
 * for all elements of A + B.
 *
 * @tparam Ta type of elements in matrix A
 * @tparam R number of rows (common to A and B)
 * @tparam C number of columns (common to A and B)
 */

template <typename Ta, int R, int C>
class add_mat_vari<Ta, double, R, C> : public vari {
 public:
  int rows_;
  int cols_;
  int size_;
  double* Ad_;
  double* Bd_;
  vari** variRefA_;
  vari** variRefAplusB_;

  /**
   * Constructor for add_mat_vari.
   *
   * All memeory is allocated in ChainableStack's
   * stack_alloc arena.
   *
   * It is critical for the efficiency of this object
   * that the constructor create new varis that aren't
   * popped onto the var_stack_, but rather are
   * popped onto the var_nochain_stack_. This is
   * controlled by the second argument to
   * vari's constructor.
   *
   * @param A matrix
   * @param  B matrix
   **/
  add_mat_vari(const Eigen::Matrix<Ta, R, C>& A,
               const Eigen::Matrix<double, R, C>& B)
      : vari(0.0),
        rows_(A.rows()),
        cols_(B.cols()),
        size_(A.size()),
        Ad_(ChainableStack::instance_->memalloc_.alloc_array<double>(size_)),
        Bd_(ChainableStack::instance_->memalloc_.alloc_array<double>(size_)),
        variRefA_(
            ChainableStack::instance_->memalloc_.alloc_array<vari*>(size_)),
        variRefAplusB_(
            ChainableStack::instance_->memalloc_.alloc_array<vari*>(size_)) {
    using Eigen::Map;
    Map<matrix_vi>(variRefA_, rows_, cols_) = A.vi();
    Map<matrix_d> Ad(Ad_, rows_, cols_);
    Map<matrix_d> Bd(Bd_, rows_, cols_);
    Ad = A.val();
    Bd = B;
    Map<matrix_vi>(variRefAplusB_, rows_, cols_)
        = (Ad + Bd).unaryExpr([](double x) { return new vari(x, false); });
  }

  virtual void chain() {
    using Eigen::Map;
    matrix_d adjAplusB(rows_, cols_);
    adjAplusB = Map<matrix_vi>(variRefAplusB_, rows_, cols_).adj();
    Map<matrix_vi>(variRefA_, rows_, cols_).adj() += adjAplusB;
  }
};

/**
 * Return the sum of two matrices
 * @tparam Mat1 type of first matrix
 * @tparam Mat2 type of second matrix
 *
 * @param[in] A matrix
 * @param[in] B matrix
 * @return Sum of two matrices
 */
template <typename Mat1, typename Mat2,
          require_all_eigen_t<Mat1, Mat2>* = nullptr,
          require_any_eigen_vt<is_var, Mat1, Mat2>* = nullptr>
auto add(const Mat1& A, const Mat2& B) {
  using Ta = value_type_t<Mat1>;
  using Tb = value_type_t<Mat2>;
  constexpr int R = Mat1::RowsAtCompileTime;
  constexpr int C = Mat1::ColsAtCompileTime;
  check_matching_dims("add", "A", A, "B", B);
  check_not_nan("add", "A", A);
  check_not_nan("add", "B", B);

  // Manage memory in arena allocator
  add_mat_vari<Ta, Tb, R, C>* baseVari = new add_mat_vari<Ta, Tb, R, C>(A, B);
  Eigen::Matrix<var, R, C> AplusB_v(A.rows(), A.cols());
  AplusB_v.vi()
      = Eigen::Map<matrix_vi>(&baseVari->variRefAplusB_[0], A.rows(), A.cols());

  return AplusB_v;
}

/**
 * This is a subclass of the vari class for matrix
 * addition A + c, where A is a N by M matrix and
 * c is a scalar.
 *
 * The class stores the structure of each matrix,
 * the double values of A and c, and pointers to
 * the varis for A and c if A or c is a var. It
 * also instatiates and stores pointers to varis
 * for all elements of A + c.
 *
 * @tparam Ta type of elements in matrix A
 * @tparam Tc type of elements in matrix B
 * @tparam R number of rows (common to A and B)
 * @tparam C number of columns (common to A and B)
 */

template <typename Ta, typename Tc, int R, int C>
class add_mat_scal_vari : public vari {
 public:
  int rows_;
  int cols_;
  int size_;
  double* Ad_;
  double cd_;
  vari** variRefA_;
  vari* variRefc_;
  vari** variRefAplusc_;

  /**
   * Constructor for add_mat_vari.
   *
   * All memory is allocated in ChainableStack's
   * stack_alloc arena.
   *
   * It is critical for the efficiency of this object
   * that the constructor create new varis that aren't
   * popped onto the var_stack_, but rather are
   * popped onto the var_nochain_stack_. This is
   * controlled by the second argument to
   * vari's constructor.
   *
   * @param A matrix
   * @param c scalar
   **/
  add_mat_scal_vari(const Eigen::Matrix<Ta, R, C>& A, const Tc c)
      : vari(0.0),
        rows_(A.rows()),
        cols_(A.cols()),
        size_(A.size()),
        Ad_(ChainableStack::instance_->memalloc_.alloc_array<double>(size_)),
        cd_(c.val()),
        variRefA_(
            ChainableStack::instance_->memalloc_.alloc_array<vari*>(size_)),
        variRefc_(c.vi_),
        variRefAplusc_(
            ChainableStack::instance_->memalloc_.alloc_array<vari*>(size_)) {
    using Eigen::Map;
    Map<matrix_vi>(variRefA_, rows_, cols_) = A.vi();
    Map<matrix_d> Ad(Ad_, rows_, cols_);
    double cd = c.val();
    Ad = A.val();
    Map<matrix_vi>(variRefAplusc_, rows_, cols_)
        = (Ad.array() + cd).matrix().eval().unaryExpr([](double x) {
            return new vari(x, false);
          });
  }

  virtual void chain() {
    using Eigen::Map;
    matrix_d adjAplusc(rows_, cols_);
    adjAplusc = Map<matrix_vi>(variRefAplusc_, rows_, cols_).adj();
    Map<matrix_vi>(variRefA_, rows_, cols_).adj() += adjAplusc;
    variRefc_->adj_ += adjAplusc.sum();
  }
};

/**
 * This is a subclass of the vari class for matrix
 * addition A + c, where A is a N by M matrix and
 * c is a double.
 *
 * The class stores the structure of each matrix,
 * the double values of A and c, and pointers to
 * the varis for A and c if A or c is a var. It
 * also instatiates and stores pointers to varis
 * for all elements of A + c.
 *
 * @tparam Ta type of elements in matrix A
 * @tparam Tc type of elements in matrix B
 * @tparam R number of rows (common to A and B)
 * @tparam C number of columns (common to A and B)
 */

template <typename Ta, int R, int C>
class add_mat_scal_vari<Ta, double, R, C> : public vari {
 public:
  int rows_;
  int cols_;
  int size_;
  double* Ad_;
  double cd_;
  vari** variRefA_;
  vari** variRefAplusc_;

  /**
   * Constructor for add_mat_vari.
   *
   * All memory is allocated in ChainableStack's
   * stack_alloc arena.
   *
   * It is critical for the efficiency of this object
   * that the constructor create new varis that aren't
   * popped onto the var_stack_, but rather are
   * popped onto the var_nochain_stack_. This is
   * controlled by the second argument to
   * vari's constructor.
   *
   * @param A matrix
   * @param c scalar
   **/
  add_mat_scal_vari(const Eigen::Matrix<Ta, R, C>& A, const double c)
      : vari(0.0),
        rows_(A.rows()),
        cols_(A.cols()),
        size_(A.size()),
        Ad_(ChainableStack::instance_->memalloc_.alloc_array<double>(size_)),
        cd_(c),
        variRefA_(
            ChainableStack::instance_->memalloc_.alloc_array<vari*>(size_)),
        variRefAplusc_(
            ChainableStack::instance_->memalloc_.alloc_array<vari*>(size_)) {
    using Eigen::Map;
    Map<matrix_vi>(variRefA_, rows_, cols_) = A.vi();
    Map<matrix_d> Ad(Ad_, rows_, cols_);
    double cd = c;
    Ad = A.val();
    Map<matrix_vi>(variRefAplusc_, rows_, cols_)
        = (Ad.array() + cd).matrix().eval().unaryExpr([](double x) {
            return new vari(x, false);
          });
  }

  virtual void chain() {
    using Eigen::Map;
    matrix_d adjAplusc(rows_, cols_);
    adjAplusc = Map<matrix_vi>(variRefAplusc_, rows_, cols_).adj();
    Map<matrix_vi>(variRefA_, rows_, cols_).adj() += adjAplusc;
  }
};

/**
 * This is a subclass of the vari class for matrix
 * addition A + c, where A is a N by M matrix of
 * doubles c is a scalar var.
 *
 * The class stores the structure of each matrix,
 * the double values of A and c, and pointers to
 * the varis for A and c if A or c is a var. It
 * also instatiates and stores pointers to varis
 * for all elements of A + c.
 *
 * @tparam Ta type of elements in matrix A
 * @tparam Tc type of elements in matrix B
 * @tparam R number of rows (common to A and B)
 * @tparam C number of columns (common to A and B)
 */

template <typename Tc, int R, int C>
class add_mat_scal_vari<double, Tc, R, C> : public vari {
 public:
  int rows_;
  int cols_;
  int size_;
  double* Ad_;
  double cd_;
  vari* variRefc_;
  vari** variRefAplusc_;

  /**
   * Constructor for add_mat_vari.
   *
   * All memory is allocated in ChainableStack's
   * stack_alloc arena.
   *
   * It is critical for the efficiency of this object
   * that the constructor create new varis that aren't
   * popped onto the var_stack_, but rather are
   * popped onto the var_nochain_stack_. This is
   * controlled by the second argument to
   * vari's constructor.
   *
   * @param A matrix
   * @param c scalar
   **/
  add_mat_scal_vari(const Eigen::Matrix<double, R, C>& A, const Tc c)
      : vari(0.0),
        rows_(A.rows()),
        cols_(A.cols()),
        size_(A.size()),
        Ad_(ChainableStack::instance_->memalloc_.alloc_array<double>(size_)),
        cd_(c.val()),
        variRefc_(c.vi_),
        variRefAplusc_(
            ChainableStack::instance_->memalloc_.alloc_array<vari*>(size_)) {
    using Eigen::Map;

    double cd = c.val();
    Map<matrix_d> Ad(Ad_, rows_, cols_);
    Ad = A.val();
    Map<matrix_vi>(variRefAplusc_, rows_, cols_)
        = (Ad.array() + cd).matrix().unaryExpr([](double x) {
            return new vari(x, false);
          });
  }

  virtual void chain() {
    using Eigen::Map;
    matrix_d adjAplusc(rows_, cols_);
    adjAplusc = Map<matrix_vi>(variRefAplusc_, rows_, cols_).adj();
    variRefc_->adj_ += adjAplusc.sum();
  }
};

/**
 * Return the sum of the specified matrix and specified scalar.
 *
 * @tparam Mat type of the matrix or expression
 * @tparam Scal type of the scalar
 * @param m Matrix or expression.
 * @param c Scalar.
 * @return The matrix plus the scalar.
 */
template <typename Mat, typename Scal, require_eigen_t<Mat>* = nullptr,
          require_stan_scalar_t<Scal>* = nullptr,
          require_any_var_t<typename value_type<Mat>::type, Scal>* = nullptr>
auto add(const Mat& A, const Scal c) {
  using Ta = value_type_t<Mat>;
  constexpr int R = Mat::RowsAtCompileTime;
  constexpr int C = Mat::ColsAtCompileTime;
  check_not_nan("add matrix-scalar A + c", "A", A);
  check_not_nan("add matrix-scalar A + c", "c", c);

  add_mat_scal_vari<Ta, Scal, R, C>* baseVari
      = new add_mat_scal_vari<Ta, Scal, R, C>(A, c);
  Eigen::Matrix<var, R, C> Aplusc_v(A.rows(), A.cols());
  Aplusc_v.vi()
      = Eigen::Map<matrix_vi>(&baseVari->variRefAplusc_[0], A.rows(), A.cols());

  return Aplusc_v;
}

/**
 * Return the sum of the specified scalar and specified matrix.
 *
 * @tparam Scal type of the scalar
 * @tparam Mat type of the matrix or expression
 * @param c Scalar.
 * @param m Matrix.
 * @return The scalar plus the matrix.
 */
template <typename Mat, typename Scal, require_eigen_t<Mat>* = nullptr,
          require_stan_scalar_t<Scal>* = nullptr,
          require_any_var_t<typename value_type<Mat>::type, Scal>* = nullptr>
inline auto add(const Scal c, const Mat& A) {
  return add(A, c);
}

}  // namespace math
}  // namespace stan

#endif
