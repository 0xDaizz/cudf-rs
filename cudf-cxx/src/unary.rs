//! Bridge definitions for libcudf unary operations.
//!
//! Provides unary transformations (math, null checks, type casting)
//! on GPU-resident columns.

#[cxx::bridge(namespace = "cudf_shims")]
pub mod ffi {
    unsafe extern "C++" {
        include!("unary_shim.h");
        include!("column_shim.h");

        type OwnedColumn = crate::column::ffi::OwnedColumn;

        // ── Unary Operations ──────────────────────────────────────

        /// Apply a unary operation to a column.
        /// `op` is the cudf::unary_operator enum value.
        fn unary_operation(input: &OwnedColumn, op: i32) -> Result<UniquePtr<OwnedColumn>>;

        /// Return a bool8 column indicating which elements are null.
        fn is_null(input: &OwnedColumn) -> Result<UniquePtr<OwnedColumn>>;

        /// Return a bool8 column indicating which elements are valid (non-null).
        fn is_valid(input: &OwnedColumn) -> Result<UniquePtr<OwnedColumn>>;

        /// Return a bool8 column indicating which elements are NaN.
        fn is_nan(input: &OwnedColumn) -> Result<UniquePtr<OwnedColumn>>;

        /// Return a bool8 column indicating which elements are not NaN.
        fn is_not_nan(input: &OwnedColumn) -> Result<UniquePtr<OwnedColumn>>;

        /// Cast a column to a different data type.
        fn cast(input: &OwnedColumn, type_id: i32) -> Result<UniquePtr<OwnedColumn>>;

        /// Check if a cast between two data types is supported.
        fn is_supported_cast(from_type_id: i32, to_type_id: i32) -> bool;
    }
}
