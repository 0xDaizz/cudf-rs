//! Bridge definitions for libcudf string split operations.

#[cxx::bridge(namespace = "cudf_shims")]
pub mod ffi {
    unsafe extern "C++" {
        include!("strings/split_shim.h");
        include!("column_shim.h");
        include!("table_shim.h");

        type OwnedColumn = crate::column::ffi::OwnedColumn;
        type OwnedTable = crate::table::ffi::OwnedTable;

        /// Split each string by the delimiter, returning a table of string columns.
        /// Each row produces one element per resulting column.
        fn str_split(
            col: &OwnedColumn,
            delimiter: &str,
            maxsplit: i32,
        ) -> Result<UniquePtr<OwnedTable>>;

        /// Split each string by the delimiter from the right.
        fn str_rsplit(
            col: &OwnedColumn,
            delimiter: &str,
            maxsplit: i32,
        ) -> Result<UniquePtr<OwnedTable>>;

        /// Split each string, returning a list column of strings per row.
        fn str_split_record(
            col: &OwnedColumn,
            delimiter: &str,
            maxsplit: i32,
        ) -> Result<UniquePtr<OwnedColumn>>;

        /// Split each string from the right, returning a list column.
        fn str_rsplit_record(
            col: &OwnedColumn,
            delimiter: &str,
            maxsplit: i32,
        ) -> Result<UniquePtr<OwnedColumn>>;
    }
}
