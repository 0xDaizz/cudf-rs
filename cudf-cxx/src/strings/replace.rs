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

        /// Replace characters in the [start, stop) range with `replacement`.
        fn str_replace_slice(
            col: &OwnedColumn,
            replacement: &str,
            start: i32,
            stop: i32,
        ) -> Result<UniquePtr<OwnedColumn>>;

        /// Replace multiple target strings with corresponding replacements.
        fn str_replace_multiple(
            col: &OwnedColumn,
            targets: &OwnedColumn,
            replacements: &OwnedColumn,
        ) -> Result<UniquePtr<OwnedColumn>>;

        /// Replace regex matches using back-references in the replacement template.
        fn str_replace_with_backrefs(
            col: &OwnedColumn,
            pattern: &str,
            replacement: &str,
        ) -> Result<UniquePtr<OwnedColumn>>;

        /// Replace multiple regex patterns with corresponding replacements from a column.
        fn str_replace_re_multiple(
            col: &OwnedColumn,
            patterns: &[String],
            replacements: &OwnedColumn,
        ) -> Result<UniquePtr<OwnedColumn>>;
    }
}
