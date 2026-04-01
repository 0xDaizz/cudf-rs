//! Bridge definitions for libcudf stream compaction operations.
//!
//! Provides GPU-accelerated null dropping, boolean masking, and
//! duplicate removal for tables and columns.

#[cxx::bridge(namespace = "cudf_shims")]
pub mod ffi {
    unsafe extern "C++" {
        include!("stream_compaction_shim.h");
        include!("column_shim.h");
        include!("table_shim.h");

        type OwnedColumn = crate::column::ffi::OwnedColumn;
        type OwnedTable = crate::table::ffi::OwnedTable;

        // ── Null Dropping ──────────────────────────────────────────

        /// Drop rows from a table where key columns contain nulls.
        /// `keys` are column indices to check. `threshold` is min non-null count to keep a row.
        fn drop_nulls_table(
            table: &OwnedTable,
            keys: &[i32],
            threshold: i32,
        ) -> Result<UniquePtr<OwnedTable>>;

        /// Drop null values from a single column.
        fn drop_nulls_column(col: &OwnedColumn) -> Result<UniquePtr<OwnedColumn>>;

        /// Drop rows from a table where key columns contain NaN.
        fn drop_nans(table: &OwnedTable, keys: &[i32]) -> Result<UniquePtr<OwnedTable>>;

        /// Drop rows from a table where key columns contain NaN, with a threshold.
        fn drop_nans_threshold(
            table: &OwnedTable,
            keys: &[i32],
            threshold: i32,
        ) -> Result<UniquePtr<OwnedTable>>;

        // ── Boolean Mask ───────────────────────────────────────────

        /// Keep only rows where boolean_mask is true.
        fn apply_boolean_mask(
            table: &OwnedTable,
            boolean_mask: &OwnedColumn,
        ) -> Result<UniquePtr<OwnedTable>>;

        // ── Deduplication ──────────────────────────────────────────

        /// Return unique rows based on key columns.
        /// `keep`: 0=ANY, 1=FIRST, 2=LAST, 3=NONE.
        /// `null_equality`: 0=EQUAL, 1=UNEQUAL.
        fn unique(
            table: &OwnedTable,
            keys: &[i32],
            keep: i32,
            null_equality: i32,
        ) -> Result<UniquePtr<OwnedTable>>;

        /// Return distinct rows based on key columns.
        fn distinct(
            table: &OwnedTable,
            keys: &[i32],
            keep: i32,
            null_equality: i32,
        ) -> Result<UniquePtr<OwnedTable>>;

        /// Count the number of distinct elements in a column.
        /// `null_handling`: 0=EXCLUDE, 1=INCLUDE.
        /// `nan_handling`: 0=NAN_IS_VALID, 1=NAN_IS_NULL.
        fn distinct_count_column(
            col: &OwnedColumn,
            null_handling: i32,
            nan_handling: i32,
        ) -> Result<i32>;

        /// Return indices of distinct rows in a table.
        fn distinct_indices(
            table: &OwnedTable,
            keep: i32,
            null_equality: i32,
        ) -> Result<UniquePtr<OwnedColumn>>;

        /// Return distinct rows preserving input order.
        fn stable_distinct(
            table: &OwnedTable,
            keys: &[i32],
            keep: i32,
            null_equality: i32,
        ) -> Result<UniquePtr<OwnedTable>>;

        /// Count consecutive groups of equivalent rows in a column.
        fn unique_count_column(
            col: &OwnedColumn,
            null_handling: i32,
            nan_handling: i32,
        ) -> Result<i32>;

        /// Count consecutive groups of equivalent rows in a table.
        fn unique_count_table(table: &OwnedTable, null_equality: i32) -> Result<i32>;

        /// Count distinct rows in a table.
        fn distinct_count_table(table: &OwnedTable, null_equality: i32) -> Result<i32>;
    }
}
