//! Bridge definitions for libcudf rounding operations.
//!
//! Provides GPU-accelerated rounding of numeric columns.

#[cxx::bridge(namespace = "cudf_shims")]
pub mod ffi {
    unsafe extern "C++" {
        include!("round_shim.h");
        include!("column_shim.h");
        type OwnedColumn = crate::column::ffi::OwnedColumn;

        /// Round a numeric column to the specified number of decimal places.
        fn round_column(col: &OwnedColumn, decimal_places: i32) -> Result<UniquePtr<OwnedColumn>>;
    }
}
