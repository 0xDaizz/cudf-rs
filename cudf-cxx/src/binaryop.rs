//! Bridge definitions for libcudf binary operations.
//!
//! Provides element-wise binary operations (arithmetic, comparison,
//! logical, bitwise) between GPU-resident columns and/or scalars.

#[cxx::bridge(namespace = "cudf_shims")]
pub mod ffi {
    unsafe extern "C++" {
        include!("binaryop_shim.h");
        include!("column_shim.h");
        include!("scalar_shim.h");

        type OwnedColumn = crate::column::ffi::OwnedColumn;
        type OwnedScalar = crate::scalar::ffi::OwnedScalar;

        // ── Binary Operations ─────────────────────────────────────

        /// Binary operation: column op column.
        /// `op` is the cudf::binary_operator enum value.
        /// `output_type` is the cudf::type_id of the result column.
        fn binary_operation_col_col(
            lhs: &OwnedColumn,
            rhs: &OwnedColumn,
            op: i32,
            output_type: i32,
        ) -> Result<UniquePtr<OwnedColumn>>;

        /// Binary operation: column op scalar.
        fn binary_operation_col_scalar(
            lhs: &OwnedColumn,
            rhs: &OwnedScalar,
            op: i32,
            output_type: i32,
        ) -> Result<UniquePtr<OwnedColumn>>;

        /// Binary operation: scalar op column.
        fn binary_operation_scalar_col(
            lhs: &OwnedScalar,
            rhs: &OwnedColumn,
            op: i32,
            output_type: i32,
        ) -> Result<UniquePtr<OwnedColumn>>;

        /// Check if a binary operation is supported for the given types.
        fn is_supported_operation(
            out_type: i32,
            lhs_type: i32,
            rhs_type: i32,
            op: i32,
        ) -> Result<bool>;
    }
}
