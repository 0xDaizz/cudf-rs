//! Bridge definitions for libcudf partitioning operations.
//!
//! Provides GPU-accelerated hash and round-robin partitioning of tables.

#[cxx::bridge(namespace = "cudf_shims")]
pub mod ffi {
    unsafe extern "C++" {
        include!("partitioning_shim.h");
        include!("table_shim.h");
        include!("column_shim.h");
        type OwnedTable = crate::table::ffi::OwnedTable;
        type OwnedColumn = crate::column::ffi::OwnedColumn;

        /// Partition a table by hashing the specified columns.
        /// Returns a reordered table (rows grouped by partition).
        fn hash_partition(
            table: &OwnedTable,
            columns_to_hash: &[i32],
            num_partitions: i32,
        ) -> Result<UniquePtr<OwnedTable>>;

        /// Partition a table using round-robin assignment.
        fn round_robin_partition(
            table: &OwnedTable,
            num_partitions: i32,
        ) -> Result<UniquePtr<OwnedTable>>;

        // ── Partition by map ──────────────────────────────────────

        /// Opaque result type for partition (table + offsets).
        type PartitionResult;

        fn num_offsets(self: &PartitionResult) -> i32;
        fn get_offset(self: &PartitionResult, index: i32) -> i32;

        /// Partition a table using a partition map column.
        fn partition(
            table: &OwnedTable,
            partition_map: &OwnedColumn,
            num_partitions: i32,
        ) -> Result<UniquePtr<PartitionResult>>;

        /// Extract the table from a PartitionResult.
        fn partition_result_table(
            result: UniquePtr<PartitionResult>,
        ) -> Result<UniquePtr<OwnedTable>>;

        /// Get the offsets from a PartitionResult.
        fn partition_result_offsets(result: &PartitionResult) -> Vec<i32>;
    }
}
