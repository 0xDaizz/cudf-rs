//! Bridge definitions for libcudf regex-based string split operations.

#[cxx::bridge(namespace = "cudf_shims")]
pub mod ffi {
    unsafe extern "C++" {
        include!("strings/split_re_shim.h");
        include!("column_shim.h");
        include!("table_shim.h");

        type OwnedColumn = crate::column::ffi::OwnedColumn;
        type OwnedTable = crate::table::ffi::OwnedTable;

        /// Split each string by regex `pattern`, returning a table of string columns.
        fn str_split_re(
            col: &OwnedColumn,
            pattern: &str,
            maxsplit: i32,
        ) -> Result<UniquePtr<OwnedTable>>;

        /// Split each string by regex `pattern` from the right.
        fn str_rsplit_re(
            col: &OwnedColumn,
            pattern: &str,
            maxsplit: i32,
        ) -> Result<UniquePtr<OwnedTable>>;

        /// Split each string by regex, returning a list column of strings per row.
        fn str_split_record_re(
            col: &OwnedColumn,
            pattern: &str,
            maxsplit: i32,
        ) -> Result<UniquePtr<OwnedColumn>>;

        /// Split each string by regex from the right, returning a list column.
        fn str_rsplit_record_re(
            col: &OwnedColumn,
            pattern: &str,
            maxsplit: i32,
        ) -> Result<UniquePtr<OwnedColumn>>;
    }
}
