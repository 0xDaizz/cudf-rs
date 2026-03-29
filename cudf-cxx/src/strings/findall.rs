//! Bridge definitions for libcudf string findall operations.

#[cxx::bridge(namespace = "cudf_shims")]
pub mod ffi {
    unsafe extern "C++" {
        include!("strings/findall_shim.h");
        include!("column_shim.h");

        type OwnedColumn = crate::column::ffi::OwnedColumn;

        /// Find all occurrences of `pattern` in each string.
        /// Returns a lists column of strings.
        fn str_findall(col: &OwnedColumn, pattern: &str) -> Result<UniquePtr<OwnedColumn>>;

        /// Find starting position of first regex match in each string.
        /// Returns an INT32 column (-1 if not found).
        fn str_find_re(col: &OwnedColumn, pattern: &str) -> Result<UniquePtr<OwnedColumn>>;
    }
}
