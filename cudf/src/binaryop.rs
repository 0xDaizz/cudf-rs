//! Binary operations on GPU-resident columns.
//!
//! Provides element-wise binary operations (arithmetic, comparison,
//! logical, bitwise) between [`Column`]s and/or [`Scalar`]s.
//!
//! # Examples
//!
//! ```rust,no_run
//! use cudf::{Column, DataType, TypeId};
//! use cudf::binaryop::BinaryOp;
//!
//! let a = Column::from_slice(&[1i32, 2, 3]).unwrap();
//! let b = Column::from_slice(&[10i32, 20, 30]).unwrap();
//! let sum = a.binary_op(&b, BinaryOp::Add, DataType::new(TypeId::Int32)).unwrap();
//! ```

use crate::column::Column;
use crate::error::{CudfError, Result};
use crate::scalar::Scalar;
use crate::types::DataType;

/// Binary operations supported by libcudf.
///
/// These map to `cudf::binary_operator` enum values in libcudf 26.x.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum BinaryOp {
    /// Addition
    Add = 0,
    /// Subtraction
    Sub = 1,
    /// Multiplication
    Mul = 2,
    /// Division (integral division for integer types)
    Div = 3,
    /// True division (always produces floating-point result)
    TrueDiv = 4,
    /// Floor division
    FloorDiv = 5,
    /// Modulo
    Mod = 6,
    /// Positive modulo (result is always >= 0)
    PMod = 7,
    /// Python-style modulo (result has sign of divisor)
    PyMod = 8,
    /// Power
    Pow = 9,
    /// Integer power
    IntPow = 10,
    /// Logarithm with specified base
    LogBase = 11,
    /// Two-argument arctangent
    Atan2 = 12,
    /// Shift left
    ShiftLeft = 13,
    /// Shift right
    ShiftRight = 14,
    /// Unsigned shift right
    ShiftRightUnsigned = 15,
    /// Bitwise AND
    BitwiseAnd = 16,
    /// Bitwise OR
    BitwiseOr = 17,
    /// Bitwise XOR
    BitwiseXor = 18,
    /// Logical AND
    LogicalAnd = 19,
    /// Logical OR
    LogicalOr = 20,
    /// Equality comparison
    Equal = 21,
    /// Inequality comparison
    NotEqual = 22,
    /// Less than
    Less = 23,
    /// Greater than
    Greater = 24,
    /// Less than or equal
    LessEqual = 25,
    /// Greater than or equal
    GreaterEqual = 26,
    /// Null-aware equality (NULL == NULL -> true)
    NullEquals = 27,
    /// Null-aware inequality
    NullNotEquals = 28,
    /// Null-aware max
    NullMax = 29,
    /// Null-aware min
    NullMin = 30,
    /// Generic binary operation (user-defined)
    GenericBinary = 31,
    /// Null-aware logical AND
    NullLogicalAnd = 32,
    /// Null-aware logical OR
    NullLogicalOr = 33,
    /// Invalid binary operator (sentinel)
    InvalidBinary = 34,
}

impl Column {
    /// Apply a binary operation element-wise between two columns.
    ///
    /// Both columns must have the same length.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use cudf::{Column, DataType, TypeId};
    /// use cudf::binaryop::BinaryOp;
    ///
    /// let a = Column::from_slice(&[1i32, 2, 3]).unwrap();
    /// let b = Column::from_slice(&[10i32, 20, 30]).unwrap();
    /// let result = a.binary_op(&b, BinaryOp::Add, DataType::new(TypeId::Int32)).unwrap();
    /// ```
    pub fn binary_op(&self, other: &Column, op: BinaryOp, output_type: DataType) -> Result<Column> {
        let result = cudf_cxx::binaryop::ffi::binary_operation_col_col(
            &self.inner,
            &other.inner,
            op as i32,
            output_type.id() as i32,
        ).map_err(CudfError::from_cxx)?;
        Ok(Column { inner: result })
    }

    /// Apply a binary operation element-wise between this column and a scalar.
    ///
    /// The scalar is broadcast to match the column length.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use cudf::{Column, Scalar, DataType, TypeId};
    /// use cudf::binaryop::BinaryOp;
    ///
    /// let col = Column::from_slice(&[1i32, 2, 3]).unwrap();
    /// let scalar = Scalar::new(10i32).unwrap();
    /// let result = col.binary_op_scalar(&scalar, BinaryOp::Mul, DataType::new(TypeId::Int32)).unwrap();
    /// ```
    pub fn binary_op_scalar(&self, scalar: &Scalar, op: BinaryOp, output_type: DataType) -> Result<Column> {
        let result = cudf_cxx::binaryop::ffi::binary_operation_col_scalar(
            &self.inner,
            &scalar.inner,
            op as i32,
            output_type.id() as i32,
        ).map_err(CudfError::from_cxx)?;
        Ok(Column { inner: result })
    }
}

/// Apply a binary operation between a scalar and a column.
///
/// The scalar is broadcast to match the column length.
pub fn binary_op(lhs: &Scalar, rhs: &Column, op: BinaryOp, output_type: DataType) -> Result<Column> {
    let result = cudf_cxx::binaryop::ffi::binary_operation_scalar_col(
        &lhs.inner,
        &rhs.inner,
        op as i32,
        output_type.id() as i32,
    ).map_err(CudfError::from_cxx)?;
    Ok(Column { inner: result })
}
