//! Bridge definitions for libcudf string attributes operations.

#[cxx::bridge(namespace = "cudf_shims")]
pub mod ffi {
    unsafe extern "C++" {
        include!("strings/attributes_shim.h");
        include!("column_shim.h");

        type OwnedColumn = crate::column::ffi::OwnedColumn;

        /// Count the number of characters in each string.
        fn str_count_characters(col: &OwnedColumn) -> Result<UniquePtr<OwnedColumn>>;

        /// Count the number of bytes in each string.
        fn str_count_bytes(col: &OwnedColumn) -> Result<UniquePtr<OwnedColumn>>;

        /// Return the code points for each character of each string.
        fn str_code_points(col: &OwnedColumn) -> Result<UniquePtr<OwnedColumn>>;
    }
}
