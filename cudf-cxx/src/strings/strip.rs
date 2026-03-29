//! Bridge definitions for libcudf string strip (trim) operations.

#[cxx::bridge(namespace = "cudf_shims")]
pub mod ffi {
    unsafe extern "C++" {
        include!("strings/strip_shim.h");
        include!("column_shim.h");

        type OwnedColumn = crate::column::ffi::OwnedColumn;

        /// Strip leading and trailing whitespace from each string.
        fn str_strip(col: &OwnedColumn) -> Result<UniquePtr<OwnedColumn>>;

        /// Strip leading whitespace from each string.
        fn str_lstrip(col: &OwnedColumn) -> Result<UniquePtr<OwnedColumn>>;

        /// Strip trailing whitespace from each string.
        fn str_rstrip(col: &OwnedColumn) -> Result<UniquePtr<OwnedColumn>>;
    }
}
