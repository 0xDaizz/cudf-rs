//! Bridge definitions for Arrow interop operations.
//!
//! Provides serialization of columns and tables to/from Arrow IPC format,
//! enabling zero-copy-ish interop with the Arrow ecosystem.

#[cxx::bridge(namespace = "cudf_shims")]
pub mod ffi {
    unsafe extern "C++" {
        include!("interop_shim.h");
        include!("column_shim.h");
        include!("table_shim.h");
        type OwnedColumn = crate::column::ffi::OwnedColumn;
        type OwnedTable = crate::table::ffi::OwnedTable;

        /// Export a column to Arrow IPC format (serialized bytes).
        fn column_to_arrow_ipc(col: &OwnedColumn) -> Result<Vec<u8>>;

        /// Import a column from Arrow IPC format.
        fn column_from_arrow_ipc(data: &[u8]) -> Result<UniquePtr<OwnedColumn>>;

        /// Export a table to Arrow IPC format.
        fn table_to_arrow_ipc(table: &OwnedTable) -> Result<Vec<u8>>;

        /// Import a table from Arrow IPC format.
        fn table_from_arrow_ipc(data: &[u8]) -> Result<UniquePtr<OwnedTable>>;
    }
}
