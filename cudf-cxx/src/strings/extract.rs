//! Bridge definitions for libcudf string regex extract operations.

#[cxx::bridge(namespace = "cudf_shims")]
pub mod ffi {
    unsafe extern "C++" {
        include!("strings/extract_shim.h");
        include!("column_shim.h");
        include!("table_shim.h");

        type OwnedColumn = crate::column::ffi::OwnedColumn;
        type OwnedTable = crate::table::ffi::OwnedTable;

        /// Extract the first capture group from each string matching `pattern`.
        /// Returns a table with one column per capture group.
        fn str_extract(col: &OwnedColumn, pattern: &str) -> Result<UniquePtr<OwnedTable>>;

        /// Extract all matches of capture groups per row, returning a list column.
        fn str_extract_all_record(
            col: &OwnedColumn,
            pattern: &str,
        ) -> Result<UniquePtr<OwnedColumn>>;

        /// Extract a single capture group from each string matching `pattern`.
        fn str_extract_single(
            col: &OwnedColumn,
            pattern: &str,
            group_index: i32,
        ) -> Result<UniquePtr<OwnedColumn>>;
    }
}
