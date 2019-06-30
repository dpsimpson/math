#ifndef STAN_MATH_PRIM_MAT_MATRIX_ADDONS_H
#define STAN_MATH_PRIM_MAT_MATRIX_ADDONS_H

/**
 * Reimplements is_fvar without requiring external math headers
 *
 * decltype((void)(T::d_)) is a pre C++17 replacement for std::void_t
 */
template<class, class = void>
struct is_fvar : std::false_type
{ };
template<class T>
struct is_fvar<T, decltype((void)(T::d_))> : std::true_type
{ };

template<typename T>
using rev_rtn_type = std::conditional_t<std::is_const<typename std::remove_reference<T>::type>::value,
                                         const double&,
                                         double&>;

template<typename T>
using vi_rtn_type = std::conditional_t<std::is_const<typename std::remove_reference<T>::type>::value,
                                          const decltype(T::vi_)&,
                                          decltype(T::vi_)&>;

template<typename T>
using fwd_rtn_type = std::conditional_t<std::is_const<typename std::remove_reference<T>::type>::value,
                                         const decltype(T::val_)&,
                                         decltype(T::val_)&>;

/**
 * Structure to return a view to the values in a var, vari*, and fvar<T>.
 * To identify the correct member to call for a given input, templates
 * check a combination of whether the input is a pointer (i.e. vari*)
 * and/or whether the input has member ".d_" (i.e. fvar).
 *
 * For definitions of EIGEN_EMPTY_STRUCT_CTOR, EIGEN_DEVICE_FUNC, and
 * EIGEN_STRONG_INLINE; see: https://eigen.tuxfamily.org/dox/XprHelper_8h_source.html
 */
struct val_Op{
  EIGEN_EMPTY_STRUCT_CTOR(val_Op);

  //Returns value from a vari*
  template<typename T = Scalar>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    std::enable_if_t<std::is_pointer<T>::value, rev_rtn_type<T>>
      operator()(T &v) const { return v->val_; }

  //Returns value from a var
  template<typename T = Scalar>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    std::enable_if_t<(!std::is_pointer<T>::value && !is_fvar<T>::value),
                      rev_rtn_type<T>>
      operator()(T &v) const { return v.vi_->val_; }

  //Returns value from an fvar
  template<typename T = Scalar>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    std::enable_if_t<is_fvar<T>::value, fwd_rtn_type<T>>
      operator()(T &v) const { return v.val_; }
};

/**
 * Coefficient-wise function applying val_Op struct to a matrix of const var
 * or vari* and returning a view to the const matrix of doubles containing
 * the values
 */
inline const CwiseUnaryOp<val_Op, const Derived>
val() const { return CwiseUnaryOp<val_Op, const Derived>(derived());
}

/**
 * Coefficient-wise function applying val_Op struct to a matrix of var
 * or vari* and returning a view to the values
 */
inline CwiseUnaryView<val_Op, Derived>
val() { return CwiseUnaryView<val_Op, Derived>(derived());
}

/**
 * Structure to return tangent from an fvar.
 */
struct d_Op {
  EIGEN_EMPTY_STRUCT_CTOR(d_Op);

  //Returns tangent from an fvar
  template<typename T = Scalar>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    fwd_rtn_type<T> operator()(T &v) const { return v.d_; }
};

/**
 * Coefficient-wise function applying d__Op struct to a matrix of const fvar<T>
 * and returning a const matrix of type T containing the tangents
 */
inline const CwiseUnaryOp<d_Op, const Derived>
d() const { return CwiseUnaryOp<d_Op, const Derived>(derived());
}

/**
 * Coefficient-wise function applying d__Op struct to a matrix of fvar<T>
 * and returning a view to a matrix of type T of the tangents that can
 * be modified
 */
inline CwiseUnaryView<d_Op, Derived>
d() { return CwiseUnaryView<d_Op, Derived>(derived());
}

/**
 * Structure to return adjoints from var and vari*. Tests whether the variables
 * are pointers (i.e. vari*) to determine whether to return the adjoint or
 * first point to the underlying vari* (in the case of var).
 */
struct adj_Op {
  EIGEN_EMPTY_STRUCT_CTOR(adj_Op);

  //Returns adjoint from a vari*
  template<typename T = Scalar>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    std::enable_if_t<std::is_pointer<T>::value, rev_rtn_type<T>>
      operator()(T &v) const { return v->adj_; }

  //Returns adjoint from a var
  template<typename T = Scalar>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    std::enable_if_t<!std::is_pointer<T>::value, rev_rtn_type<T>>
      operator()(T &v) const { return v.vi_->adj_; }
};

/**
 * Coefficient-wise function applying adj_Op struct to a matrix of const var
 * and returning a const matrix of type T containing the values
 */
inline const CwiseUnaryOp<adj_Op, const Derived>
adj() const { return CwiseUnaryOp<adj_Op, const Derived>(derived());
}

/**
 * Coefficient-wise function applying adj_Op struct to a matrix of var
 * and returning a view to a matrix of doubles of the adjoints that can
 * be modified
 */
inline CwiseUnaryView<adj_Op, Derived>
adj() { return CwiseUnaryView<adj_Op, Derived>(derived());
}
/**
 * Structure to return vari* from a var.
 */
struct vi_Op {
  EIGEN_EMPTY_STRUCT_CTOR(vi_Op);

  //Returns vari* from a var
  template<typename T = Scalar>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE
    vi_rtn_type<T> operator()(T &v) const { return v.vi_; }
};

/**
 * Coefficient-wise function applying vi_Op struct to a matrix of const var
 * and returning a const matrix of vari*
 */
inline const CwiseUnaryOp<vi_Op, const Derived>
vi() const { return CwiseUnaryOp<vi_Op, const Derived>(derived());
}

/**
 * Coefficient-wise function applying vi_Op struct to a matrix of var
 * and returning a view to a matrix of vari* that can be modified
 */
inline CwiseUnaryView<vi_Op, Derived>
vi() { return CwiseUnaryView<vi_Op, Derived>(derived());
}

#endif
