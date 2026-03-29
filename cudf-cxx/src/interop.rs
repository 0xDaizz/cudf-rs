//! Bridge definitions for Arrow interop, DLPack, and contiguous_split.
//!
//! Provides:
//! - Arrow IPC serialization (legacy)
//! - Arrow C Data Interface (true zero-copy via raw pointers)
//! - DLPack tensor exchange
//! - contiguous_split / pack / unpack for efficient GPU buffer management

#[cxx::bridge(namespace = "cudf_shims")]
pub mod ffi {
    unsafe extern "C++" {
        include!("interop_shim.h");
        include!("column_shim.h");
        include!("table_shim.h");
        type OwnedColumn = crate::column::ffi::OwnedColumn;
        type OwnedTable = crate::table::ffi::OwnedTable;

        // ── Arrow IPC (legacy) ────────────────────────────────────

        /// Export a column to Arrow IPC format (serialized bytes).
        fn column_to_arrow_ipc(col: &OwnedColumn) -> Result<Vec<u8>>;

        /// Import a column from Arrow IPC format.
        fn column_from_arrow_ipc(data: &[u8]) -> Result<UniquePtr<OwnedColumn>>;

        /// Export a table to Arrow IPC format.
        fn table_to_arrow_ipc(table: &OwnedTable) -> Result<Vec<u8>>;

        /// Import a table from Arrow IPC format.
        fn table_from_arrow_ipc(data: &[u8]) -> Result<UniquePtr<OwnedTable>>;

        // ── Arrow C Data Interface ────────────────────────────────

        /// Export column schema as heap-allocated ArrowSchema (returns pointer as u64).
        fn column_to_arrow_schema_ptr(col: &OwnedColumn) -> Result<u64>;

        /// Export column data as heap-allocated ArrowArray (returns pointer as u64).
        fn column_to_arrow_array_ptr(col: &OwnedColumn) -> Result<u64>;

        /// Import a column from ArrowSchema + ArrowArray pointers.
        fn column_from_arrow_cdata(
            schema_ptr: u64,
            array_ptr: u64,
        ) -> Result<UniquePtr<OwnedColumn>>;

        /// Export table schema as heap-allocated ArrowSchema (returns pointer as u64).
        fn table_to_arrow_schema_ptr(table: &OwnedTable) -> Result<u64>;

        /// Export table data as heap-allocated ArrowArray (returns pointer as u64).
        fn table_to_arrow_array_ptr(table: &OwnedTable) -> Result<u64>;

        /// Import a table from ArrowSchema + ArrowArray pointers.
        fn table_from_arrow_cdata(schema_ptr: u64, array_ptr: u64)
        -> Result<UniquePtr<OwnedTable>>;

        /// Free an ArrowSchema without consuming it.
        fn free_arrow_schema(ptr: u64);

        /// Free an ArrowArray without consuming it.
        fn free_arrow_array(ptr: u64);

        // ── DLPack ────────────────────────────────────────────────

        /// Convert table to DLPack tensor (returns DLManagedTensor* as u64).
        fn table_to_dlpack(table: &OwnedTable) -> Result<u64>;

        /// Import table from DLPack tensor (consumes the tensor).
        fn table_from_dlpack(dlpack_ptr: u64) -> Result<UniquePtr<OwnedTable>>;

        /// Free a DLPack tensor without consuming it.
        fn free_dlpack(dlpack_ptr: u64);

        // ── contiguous_split / pack / unpack ──────────────────────

        /// Opaque handle for packed columns (cudf::packed_columns wrapper).
        type OwnedPackedColumns;

        /// Pack a table into a single contiguous GPU buffer + host metadata.
        fn pack_table(table: &OwnedTable) -> Result<UniquePtr<OwnedPackedColumns>>;

        /// Get host-side metadata bytes from packed columns.
        fn packed_metadata(packed: &OwnedPackedColumns) -> Result<Vec<u8>>;

        /// Get GPU data buffer size in bytes.
        fn packed_gpu_data_size(packed: &OwnedPackedColumns) -> Result<i64>;

        /// Unpack packed columns back into a table (deep copy).
        fn unpack_table(packed: &OwnedPackedColumns) -> Result<UniquePtr<OwnedTable>>;

        /// Perform contiguous_split.  Returns [handle, num_parts].
        fn contiguous_split_table(table: &OwnedTable, splits: &[i32]) -> Result<Vec<u64>>;

        /// Get one packed partition from a contiguous_split result.
        fn contiguous_split_get(handle: u64, index: i32) -> Result<UniquePtr<OwnedPackedColumns>>;

        /// Free a contiguous_split result handle.
        fn contiguous_split_free(handle: u64);
    }
}
