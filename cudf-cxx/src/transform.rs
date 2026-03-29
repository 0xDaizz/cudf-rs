//! Bridge definitions for libcudf transform operations.
//!
//! Provides GPU-accelerated data transformations such as NaN-to-null
//! conversion and boolean mask generation.

#[cxx::bridge(namespace = "cudf_shims")]
pub mod ffi {
    unsafe extern "C++" {
        include!("transform_shim.h");
        include!("column_shim.h");
        type OwnedColumn = crate::column::ffi::OwnedColumn;

        /// Replace NaN values with nulls in a floating-point column.
        fn nans_to_nulls(col: &OwnedColumn) -> Result<UniquePtr<OwnedColumn>>;

        /// Convert a boolean column to a bitmask (host bytes).
        fn bools_to_mask(col: &OwnedColumn) -> Result<Vec<u8>>;
    }
}
