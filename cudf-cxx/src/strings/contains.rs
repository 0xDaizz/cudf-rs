//! Bridge definitions for libcudf string contains/match operations.

#[cxx::bridge(namespace = "cudf_shims")]
pub mod ffi {
    unsafe extern "C++" {
        include!("strings/contains_shim.h");
        include!("column_shim.h");

        type OwnedColumn = crate::column::ffi::OwnedColumn;

        /// Check if each string contains the literal target.
        /// Returns a BOOL8 column.
        fn str_contains(col: &OwnedColumn, target: &str) -> Result<UniquePtr<OwnedColumn>>;

        /// Check if each string contains a match for the regex pattern.
        /// Returns a BOOL8 column.
        fn str_contains_re(col: &OwnedColumn, pattern: &str) -> Result<UniquePtr<OwnedColumn>>;

        /// Check if each string fully matches the regex pattern.
        /// Returns a BOOL8 column.
        fn str_matches_re(col: &OwnedColumn, pattern: &str) -> Result<UniquePtr<OwnedColumn>>;

        /// Count non-overlapping occurrences of the regex pattern.
        /// Returns an INT32 column.
        fn str_count_re(col: &OwnedColumn, pattern: &str) -> Result<UniquePtr<OwnedColumn>>;
    }
}
