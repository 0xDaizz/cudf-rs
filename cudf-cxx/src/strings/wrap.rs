//! Bridge definitions for libcudf string wrap operations.

#[cxx::bridge(namespace = "cudf_shims")]
pub mod ffi {
    unsafe extern "C++" {
        include!("strings/wrap_shim.h");
        include!("column_shim.h");

        type OwnedColumn = crate::column::ffi::OwnedColumn;

        /// Wrap long strings by inserting newlines at whitespace boundaries.
        fn str_wrap(col: &OwnedColumn, width: i32) -> Result<UniquePtr<OwnedColumn>>;
    }
}
