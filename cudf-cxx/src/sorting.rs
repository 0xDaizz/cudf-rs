//! Bridge definitions for libcudf sorting operations.
//!
//! Provides GPU-accelerated sorting, ranking, and order-checking functions
//! for tables and columns.

#[cxx::bridge(namespace = "cudf_shims")]
pub mod ffi {
    unsafe extern "C++" {
        include!("sorting_shim.h");
        include!("column_shim.h");
        include!("table_shim.h");

        type OwnedColumn = crate::column::ffi::OwnedColumn;
        type OwnedTable = crate::table::ffi::OwnedTable;

        // ── Sorting ────────────────────────────────────────────────

        /// Returns a column of row indices that would sort the table.
        /// `column_order`: 0=ascending, 1=descending (one per column).
        /// `null_order`: 0=before, 1=after (one per column).
        fn sorted_order(
            table: &OwnedTable,
            column_order: &[i32],
            null_order: &[i32],
        ) -> Result<UniquePtr<OwnedColumn>>;

        /// Sort a table by its columns, returning a new sorted table.
        fn sort(
            table: &OwnedTable,
            column_order: &[i32],
            null_order: &[i32],
        ) -> Result<UniquePtr<OwnedTable>>;

        /// Sort `values` table by the rows of `keys` table.
        fn sort_by_key(
            values: &OwnedTable,
            keys: &OwnedTable,
            column_order: &[i32],
            null_order: &[i32],
        ) -> Result<UniquePtr<OwnedTable>>;

        /// Stable sort `values` table by the rows of `keys` table.
        fn stable_sort_by_key(
            values: &OwnedTable,
            keys: &OwnedTable,
            column_order: &[i32],
            null_order: &[i32],
        ) -> Result<UniquePtr<OwnedTable>>;

        /// Compute the rank of each element in a column.
        /// `method`: 0=FIRST, 1=AVERAGE, 2=MIN, 3=MAX, 4=DENSE.
        /// `column_order`: 0=ascending, 1=descending.
        /// `null_order`: 0=before, 1=after.
        /// `null_handling`: 0=EXCLUDE, 1=INCLUDE.
        /// `percentage`: whether to return percentage ranks.
        fn rank(
            col: &OwnedColumn,
            method: i32,
            column_order: i32,
            null_order: i32,
            null_handling: i32,
            percentage: bool,
        ) -> Result<UniquePtr<OwnedColumn>>;

        /// Check whether a table is already sorted.
        fn is_sorted(table: &OwnedTable, column_order: &[i32], null_order: &[i32]) -> Result<bool>;

        /// Returns a column of row indices that would stably sort the table.
        fn stable_sorted_order(
            table: &OwnedTable,
            column_order: &[i32],
            null_order: &[i32],
        ) -> Result<UniquePtr<OwnedColumn>>;

        /// Stable sort a table by its columns.
        fn stable_sort(
            table: &OwnedTable,
            column_order: &[i32],
            null_order: &[i32],
        ) -> Result<UniquePtr<OwnedTable>>;

        /// Return the top k values of a column.
        /// order: 0=ascending, 1=descending.
        fn top_k(col: &OwnedColumn, k: i32, order: i32) -> Result<UniquePtr<OwnedColumn>>;

        // ── Segmented Sorting ─────────────────────────────────────

        /// Returns row indices that would sort each segment of the table.
        fn segmented_sorted_order(
            table: &OwnedTable,
            segment_offsets: &OwnedColumn,
            column_order: &[i32],
            null_order: &[i32],
        ) -> Result<UniquePtr<OwnedColumn>>;

        /// Stable version of segmented_sorted_order.
        fn stable_segmented_sorted_order(
            table: &OwnedTable,
            segment_offsets: &OwnedColumn,
            column_order: &[i32],
            null_order: &[i32],
        ) -> Result<UniquePtr<OwnedColumn>>;

        /// Sort values by keys within each segment.
        fn segmented_sort_by_key(
            values: &OwnedTable,
            keys: &OwnedTable,
            segment_offsets: &OwnedColumn,
            column_order: &[i32],
            null_order: &[i32],
        ) -> Result<UniquePtr<OwnedTable>>;

        /// Stable version of segmented_sort_by_key.
        fn stable_segmented_sort_by_key(
            values: &OwnedTable,
            keys: &OwnedTable,
            segment_offsets: &OwnedColumn,
            column_order: &[i32],
            null_order: &[i32],
        ) -> Result<UniquePtr<OwnedTable>>;
    }
}
