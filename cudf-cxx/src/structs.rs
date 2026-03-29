//! Bridge definitions for libcudf struct column operations.
//!
//! Provides operations on struct (nested) columns, such as extracting
//! child columns by index.

#[cxx::bridge(namespace = "cudf_shims")]
pub mod ffi {
    unsafe extern "C++" {
        include!("structs_shim.h");
        include!("column_shim.h");

        type OwnedColumn = crate::column::ffi::OwnedColumn;

        // ── Extract ───────────────────────────────────────────────

        /// Extract the child column at `index` from a struct column.
        /// Returns a copy of the child column.
        fn structs_extract(col: &OwnedColumn, index: i32) -> Result<UniquePtr<OwnedColumn>>;
    }
}
