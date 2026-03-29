//! Bridge definitions for libcudf string find operations.

#[cxx::bridge(namespace = "cudf_shims")]
pub mod ffi {
    unsafe extern "C++" {
        include!("strings/find_shim.h");
        include!("column_shim.h");
        include!("table_shim.h");

        type OwnedColumn = crate::column::ffi::OwnedColumn;
        type OwnedTable = crate::table::ffi::OwnedTable;

        /// Find first occurrence of target in each string, starting at `start`.
        /// Returns an INT32 column of positions (-1 if not found).
        fn str_find(col: &OwnedColumn, target: &str, start: i32) -> Result<UniquePtr<OwnedColumn>>;

        /// Find last occurrence of target in each string.
        /// Returns an INT32 column of positions (-1 if not found).
        fn str_rfind(col: &OwnedColumn, target: &str) -> Result<UniquePtr<OwnedColumn>>;

        /// Check if each string starts with the given target.
        /// Returns a BOOL8 column.
        fn str_starts_with(col: &OwnedColumn, target: &str) -> Result<UniquePtr<OwnedColumn>>;

        /// Check if each string ends with the given target.
        /// Returns a BOOL8 column.
        fn str_ends_with(col: &OwnedColumn, target: &str) -> Result<UniquePtr<OwnedColumn>>;

        /// Check if each string contains any of the target strings.
        /// Returns a table of BOOL8 columns, one per target.
        fn str_contains_multiple(
            col: &OwnedColumn,
            targets: &OwnedColumn,
        ) -> Result<UniquePtr<OwnedTable>>;
    }
}
