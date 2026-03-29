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
        fn drop_nulls_column(
            col: &OwnedColumn,
        ) -> Result<UniquePtr<OwnedColumn>>;

        /// Drop rows from a table where key columns contain NaN.
        fn drop_nans(
            table: &OwnedTable,
            keys: &[i32],
        ) -> Result<UniquePtr<OwnedTable>>;

        // ── Boolean Mask ───────────────────────────────────────────

        /// Keep only rows where boolean_mask is true.
        fn apply_boolean_mask(
            table: &OwnedTable,
            boolean_mask: &OwnedColumn,
        ) -> Result<UniquePtr<OwnedTable>>;

        // ── Deduplication ──────────────────────────────────────────

        /// Return unique rows based on key columns.
        /// `keep`: 0=FIRST, 1=LAST, 2=ANY, 3=NONE.
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
        /// `null_handling`: 0=INCLUDE, 1=EXCLUDE.
        /// `nan_handling`: 0=NAN_IS_VALID, 1=NAN_IS_NULL.
        fn distinct_count_column(
            col: &OwnedColumn,
            null_handling: i32,
            nan_handling: i32,
        ) -> Result<i32>;
    }
}
