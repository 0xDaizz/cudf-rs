//! Bridge definitions for libcudf string partition operations.

#[cxx::bridge(namespace = "cudf_shims")]
pub mod ffi {
    unsafe extern "C++" {
        include!("strings/partition_shim.h");
        include!("column_shim.h");
        include!("table_shim.h");

        type OwnedColumn = crate::column::ffi::OwnedColumn;
        type OwnedTable = crate::table::ffi::OwnedTable;

        /// Partition each string at the first occurrence of `delimiter`.
        /// Returns a 3-column table: [before, delimiter, after].
        fn str_partition(col: &OwnedColumn, delimiter: &str) -> Result<UniquePtr<OwnedTable>>;

        /// Partition each string at the last occurrence of `delimiter`.
        /// Returns a 3-column table: [before, delimiter, after].
        fn str_rpartition(col: &OwnedColumn, delimiter: &str) -> Result<UniquePtr<OwnedTable>>;
    }
}
