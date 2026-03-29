//! Bridge definitions for libcudf reshape operations.
//!
//! Provides GPU-accelerated interleaving and tiling of table columns.

#[cxx::bridge(namespace = "cudf_shims")]
pub mod ffi {
    unsafe extern "C++" {
        include!("reshape_shim.h");
        include!("table_shim.h");
        include!("column_shim.h");
        type OwnedTable = crate::table::ffi::OwnedTable;
        type OwnedColumn = crate::column::ffi::OwnedColumn;

        /// Interleave columns of a table into a single column.
        fn interleave_columns(table: &OwnedTable) -> Result<UniquePtr<OwnedColumn>>;

        /// Repeat (tile) a table's rows `count` times.
        fn tile(table: &OwnedTable, count: i32) -> Result<UniquePtr<OwnedTable>>;
    }
}
