//! Bridge definitions for libcudf string replace operations.

#[cxx::bridge(namespace = "cudf_shims")]
pub mod ffi {
    unsafe extern "C++" {
        include!("strings/replace_shim.h");
        include!("column_shim.h");

        type OwnedColumn = crate::column::ffi::OwnedColumn;

        /// Replace all occurrences of `target` with `replacement` in each string.
        fn str_replace(
            col: &OwnedColumn,
            target: &str,
            replacement: &str,
        ) -> Result<UniquePtr<OwnedColumn>>;

        /// Replace all matches of `pattern` (regex) with `replacement` in each string.
        fn str_replace_re(
            col: &OwnedColumn,
            pattern: &str,
            replacement: &str,
        ) -> Result<UniquePtr<OwnedColumn>>;
    }
}
