//! Unary operations on GPU-resident columns.
//!
//! Provides element-wise mathematical functions, null/NaN checks,
//! and type casting for [`Column`]s.
//!
//! # Examples
//!
//! ```rust,no_run
//! use cudf::{Column, DataType, TypeId};
//! use cudf::unary::UnaryOp;
//!
//! let col = Column::from_slice(&[1.0f64, 4.0, 9.0]).unwrap();
//! let result = col.unary_op(UnaryOp::Sqrt).unwrap();
//!
//! // Cast to a different type
//! let as_i32 = col.cast(DataType::new(TypeId::Int32)).unwrap();
//! ```

use crate::column::Column;
use crate::error::{CudfError, Result};
use crate::types::DataType;

/// Unary operations supported by libcudf.
///
/// These map to `cudf::unary_operator` enum values in libcudf 26.x.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum UnaryOp {
    /// Trigonometric sine
    Sin = 0,
    /// Trigonometric cosine
    Cos = 1,
    /// Trigonometric tangent
    Tan = 2,
    /// Inverse sine
    Arcsin = 3,
    /// Inverse cosine
    Arccos = 4,
    /// Inverse tangent
    Arctan = 5,
    /// Hyperbolic sine
    Sinh = 6,
    /// Hyperbolic cosine
    Cosh = 7,
    /// Hyperbolic tangent
    Tanh = 8,
    /// Inverse hyperbolic sine
    Arcsinh = 9,
    /// Inverse hyperbolic cosine
    Arccosh = 10,
    /// Inverse hyperbolic tangent
    Arctanh = 11,
    /// Exponential (e^x)
    Exp = 12,
    /// Natural logarithm
    Log = 13,
    /// Square root
    Sqrt = 14,
    /// Cube root
    Cbrt = 15,
    /// Ceiling
    Ceil = 16,
    /// Floor
    Floor = 17,
    /// Absolute value
    Abs = 18,
    /// Round to nearest integer
    Rint = 19,
    /// Count of set bits
    BitCount = 20,
    /// Bitwise invert
    BitInvert = 21,
    /// Logical not
    Not = 22,
    /// Negate
    Negate = 23,
}

impl Column {
    /// Apply a unary operation element-wise, returning a new column.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use cudf::Column;
    /// use cudf::unary::UnaryOp;
    ///
    /// let col = Column::from_slice(&[1.0f64, 4.0, 9.0]).unwrap();
    /// let sqrt = col.unary_op(UnaryOp::Sqrt).unwrap();
    /// ```
    pub fn unary_op(&self, op: UnaryOp) -> Result<Column> {
        let result = cudf_cxx::unary::ffi::unary_operation(&self.inner, op as i32)
            .map_err(CudfError::from_cxx)?;
        Ok(Column { inner: result })
    }

    /// Return a bool8 column indicating which elements are null.
    pub fn is_null(&self) -> Result<Column> {
        let result = cudf_cxx::unary::ffi::is_null(&self.inner).map_err(CudfError::from_cxx)?;
        Ok(Column { inner: result })
    }

    /// Return a bool8 column indicating which elements are valid (non-null).
    pub fn is_valid(&self) -> Result<Column> {
        let result = cudf_cxx::unary::ffi::is_valid(&self.inner).map_err(CudfError::from_cxx)?;
        Ok(Column { inner: result })
    }

    /// Return a bool8 column indicating which elements are NaN.
    ///
    /// Only applicable to floating-point columns.
    pub fn is_nan(&self) -> Result<Column> {
        let result = cudf_cxx::unary::ffi::is_nan(&self.inner).map_err(CudfError::from_cxx)?;
        Ok(Column { inner: result })
    }

    /// Return a bool8 column indicating which elements are not NaN.
    ///
    /// Only applicable to floating-point columns.
    pub fn is_not_nan(&self) -> Result<Column> {
        let result = cudf_cxx::unary::ffi::is_not_nan(&self.inner).map_err(CudfError::from_cxx)?;
        Ok(Column { inner: result })
    }

    /// Cast this column to a different data type.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use cudf::{Column, DataType, TypeId};
    ///
    /// let col = Column::from_slice(&[1.5f64, 2.7, 3.1]).unwrap();
    /// let as_i32 = col.cast(DataType::new(TypeId::Int32)).unwrap();
    /// ```
    pub fn cast(&self, dtype: DataType) -> Result<Column> {
        let result = cudf_cxx::unary::ffi::cast(&self.inner, dtype.id() as i32)
            .map_err(CudfError::from_cxx)?;
        Ok(Column { inner: result })
    }
}

/// Check if a cast between two data types is supported.
///
/// Returns `true` if casting from `from` to `to` is supported.
pub fn is_supported_cast(from: DataType, to: DataType) -> bool {
    cudf_cxx::unary::ffi::is_supported_cast(from.id() as i32, to.id() as i32)
}
