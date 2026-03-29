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
/// These map to `cudf::binary_operator` enum values.
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
    /// Python-style modulo (result has sign of divisor)
    PyMod = 7,
    /// Power
    Pow = 8,
    /// Integer power
    IntPow = 9,
    /// Logarithm with specified base
    LogBase = 10,
    /// Two-argument arctangent
    Atan2 = 11,
    /// Shift left
    ShiftLeft = 12,
    /// Shift right
    ShiftRight = 13,
    /// Unsigned shift right
    ShiftRightUnsigned = 14,
    /// Bitwise AND
    BitwiseAnd = 15,
    /// Bitwise OR
    BitwiseOr = 16,
    /// Bitwise XOR
    BitwiseXor = 17,
    /// Logical AND
    LogicalAnd = 18,
    /// Logical OR
    LogicalOr = 19,
    /// Equality comparison
    Equal = 20,
    /// Inequality comparison
    NotEqual = 21,
    /// Less than
    Less = 22,
    /// Greater than
    Greater = 23,
    /// Less than or equal
    LessEqual = 24,
    /// Greater than or equal
    GreaterEqual = 25,
    /// Null-aware equality (NULL == NULL → true)
    NullEquals = 26,
    /// Null-aware max
    NullMax = 27,
    /// Null-aware min
    NullMin = 28,
    /// Null-aware inequality
    NullNotEquals = 29,
    /// Generic binary operation (user-defined)
    GenericBinary = 30,
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
