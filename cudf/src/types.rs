//! Core type definitions for cudf.
//!
//! These types mirror libcudf's `cudf::type_id` and `cudf::data_type`,
//! providing a safe, idiomatic Rust interface for the GPU type system.

use std::fmt;

/// Identifies the element type stored in a [`Column`](crate::Column).
///
/// This enum mirrors `cudf::type_id` and covers all data types supported
/// by libcudf, including numeric, temporal, string, and nested types.
///
/// **IMPORTANT: Synchronization requirement** -- The discriminant values in
/// this enum MUST match `cudf::type_id` (C++) exactly. The cxx bridge passes
/// type IDs as `i32`, so this enum's `#[repr(i32)]` values are the source of
/// truth on the Rust side. If you add or reorder variants, update both the
/// C++ `cudf::type_id` mapping and this enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i32)]
pub enum TypeId {
    /// Empty (no data)
    Empty = 0,
    /// Signed 8-bit integer
    Int8 = 1,
    /// Signed 16-bit integer
    Int16 = 2,
    /// Signed 32-bit integer
    Int32 = 3,
    /// Signed 64-bit integer
    Int64 = 4,
    /// Unsigned 8-bit integer
    Uint8 = 5,
    /// Unsigned 16-bit integer
    Uint16 = 6,
    /// Unsigned 32-bit integer
    Uint32 = 7,
    /// Unsigned 64-bit integer
    Uint64 = 8,
    /// 32-bit floating point
    Float32 = 9,
    /// 64-bit floating point
    Float64 = 10,
    /// Boolean (stored as 8-bit)
    Bool8 = 11,
    /// Timestamp in days since epoch
    TimestampDays = 12,
    /// Timestamp in seconds since epoch
    TimestampSeconds = 13,
    /// Timestamp in milliseconds since epoch
    TimestampMilliseconds = 14,
    /// Timestamp in microseconds since epoch
    TimestampMicroseconds = 15,
    /// Timestamp in nanoseconds since epoch
    TimestampNanoseconds = 16,
    /// Duration in days
    DurationDays = 17,
    /// Duration in seconds
    DurationSeconds = 18,
    /// Duration in milliseconds
    DurationMilliseconds = 19,
    /// Duration in microseconds
    DurationMicroseconds = 20,
    /// Duration in nanoseconds
    DurationNanoseconds = 21,
    /// Dictionary-encoded column (32-bit indices)
    Dictionary32 = 22,
    /// Variable-length UTF-8 string
    String = 23,
    /// List (nested column of variable-length sequences)
    List = 24,
    /// 32-bit fixed-point decimal
    Decimal32 = 25,
    /// 64-bit fixed-point decimal
    Decimal64 = 26,
    /// 128-bit fixed-point decimal
    Decimal128 = 27,
    /// Struct (nested column of named fields)
    Struct = 28,
}

impl TypeId {
    /// Returns the size in bytes of a single element of this type.
    /// Returns 0 for variable-width types (String, List, Struct).
    pub fn size_in_bytes(self) -> usize {
        match self {
            Self::Empty => 0,
            Self::Int8 | Self::Uint8 | Self::Bool8 => 1,
            Self::Int16 | Self::Uint16 => 2,
            Self::Int32
            | Self::Uint32
            | Self::Float32
            | Self::Decimal32
            | Self::Dictionary32
            | Self::TimestampDays
            | Self::DurationDays => 4,
            Self::Int64
            | Self::Uint64
            | Self::Float64
            | Self::Decimal64
            | Self::TimestampSeconds
            | Self::TimestampMilliseconds
            | Self::TimestampMicroseconds
            | Self::TimestampNanoseconds
            | Self::DurationSeconds
            | Self::DurationMilliseconds
            | Self::DurationMicroseconds
            | Self::DurationNanoseconds => 8,
            Self::Decimal128 => 16,
            Self::String | Self::List | Self::Struct => 0,
        }
    }

    /// Whether this type has a fixed width (known size per element).
    pub fn is_fixed_width(self) -> bool {
        self.size_in_bytes() > 0
    }

    /// Whether this type is a numeric type (integer or floating point).
    pub fn is_numeric(self) -> bool {
        matches!(
            self,
            Self::Int8
                | Self::Int16
                | Self::Int32
                | Self::Int64
                | Self::Uint8
                | Self::Uint16
                | Self::Uint32
                | Self::Uint64
                | Self::Float32
                | Self::Float64
        )
    }

    /// Whether this type is an integer type.
    pub fn is_integer(self) -> bool {
        matches!(
            self,
            Self::Int8
                | Self::Int16
                | Self::Int32
                | Self::Int64
                | Self::Uint8
                | Self::Uint16
                | Self::Uint32
                | Self::Uint64
        )
    }

    /// Whether this type is a floating-point type.
    pub fn is_floating(self) -> bool {
        matches!(self, Self::Float32 | Self::Float64)
    }

    /// Whether this type is a temporal type (timestamp or duration).
    pub fn is_temporal(self) -> bool {
        matches!(
            self,
            Self::TimestampDays
                | Self::TimestampSeconds
                | Self::TimestampMilliseconds
                | Self::TimestampMicroseconds
                | Self::TimestampNanoseconds
                | Self::DurationDays
                | Self::DurationSeconds
                | Self::DurationMilliseconds
                | Self::DurationMicroseconds
                | Self::DurationNanoseconds
        )
    }

    /// Whether this type is a nested type (List, Struct).
    pub fn is_nested(self) -> bool {
        matches!(self, Self::List | Self::Struct)
    }

    /// Convert from raw i32 value. Returns None if the value is not a valid TypeId.
    pub fn from_raw(value: i32) -> Option<Self> {
        if (0..=Self::Struct as i32).contains(&value) {
            // SAFETY: TypeId is repr(i32) with contiguous values 0..=Struct
            Some(unsafe { std::mem::transmute::<i32, TypeId>(value) })
        } else {
            None
        }
    }
}

impl fmt::Display for TypeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(self, f)
    }
}

/// Describes the data type of a column, combining a [`TypeId`] with an
/// optional scale (for fixed-point decimal types).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DataType {
    id: TypeId,
    scale: i32,
}

impl DataType {
    /// Create a new DataType from a TypeId.
    pub fn new(id: TypeId) -> Self {
        Self { id, scale: 0 }
    }

    /// Create a decimal DataType with the given scale.
    ///
    /// # Errors
    /// Returns `CudfError::InvalidArgument` if `id` is not a decimal type
    /// (`Decimal32`, `Decimal64`, or `Decimal128`).
    pub fn decimal(id: TypeId, scale: i32) -> crate::error::Result<Self> {
        if !matches!(id, TypeId::Decimal32 | TypeId::Decimal64 | TypeId::Decimal128) {
            return Err(crate::error::CudfError::InvalidArgument(
                format!("decimal() requires a decimal TypeId, got {:?}", id),
            ));
        }
        Ok(Self { id, scale })
    }

    /// The type identifier.
    pub fn id(&self) -> TypeId {
        self.id
    }

    /// The scale (meaningful only for decimal types; 0 otherwise).
    pub fn scale(&self) -> i32 {
        self.scale
    }
}

impl From<TypeId> for DataType {
    fn from(id: TypeId) -> Self {
        Self::new(id)
    }
}

impl fmt::Display for DataType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.scale != 0 {
            write!(f, "{:?}(scale={})", self.id, self.scale)
        } else {
            write!(f, "{:?}", self.id)
        }
    }
}

/// Convert a `usize` value to `i32`, returning an error if it overflows.
///
/// This is used at user-facing API boundaries where sizes or indices are
/// passed to the C++ layer as `i32` (libcudf's `size_type`).
pub fn checked_i32(val: usize) -> crate::error::Result<i32> {
    i32::try_from(val).map_err(|_| {
        crate::error::CudfError::InvalidArgument(format!(
            "value {} exceeds i32::MAX ({})",
            val,
            i32::MAX
        ))
    })
}

/// Policy for handling null values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum NullHandling {
    /// Include nulls in the computation.
    Include = 0,
    /// Exclude nulls from the computation.
    Exclude = 1,
}
