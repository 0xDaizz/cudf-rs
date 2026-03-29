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
        /// `null_handling`: 0=INCLUDE, 1=EXCLUDE.
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
        fn is_sorted(
            table: &OwnedTable,
            column_order: &[i32],
            null_order: &[i32],
        ) -> Result<bool>;
    }
}
