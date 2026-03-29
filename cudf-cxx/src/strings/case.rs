//! Bridge definitions for libcudf string case operations.

#[cxx::bridge(namespace = "cudf_shims")]
pub mod ffi {
    unsafe extern "C++" {
        include!("strings/case_shim.h");
        include!("column_shim.h");

        type OwnedColumn = crate::column::ffi::OwnedColumn;

        /// Convert all characters to uppercase.
        fn str_to_upper(col: &OwnedColumn) -> Result<UniquePtr<OwnedColumn>>;

        /// Convert all characters to lowercase.
        fn str_to_lower(col: &OwnedColumn) -> Result<UniquePtr<OwnedColumn>>;

        /// Swap case of all characters.
        fn str_swapcase(col: &OwnedColumn) -> Result<UniquePtr<OwnedColumn>>;

        /// Capitalize first character of each string.
        fn str_capitalize(col: &OwnedColumn) -> Result<UniquePtr<OwnedColumn>>;

        /// Capitalize first character of each word (title case).
        fn str_title(col: &OwnedColumn) -> Result<UniquePtr<OwnedColumn>>;
    }
}
