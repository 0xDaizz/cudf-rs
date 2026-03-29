//! Bridge definitions for libcudf copying operations.
//!
//! Provides GPU-accelerated gather, scatter, slice, split, and copy operations
//! for tables and columns.

#[cxx::bridge(namespace = "cudf_shims")]
pub mod ffi {
    unsafe extern "C++" {
        include!("copying_shim.h");
        include!("column_shim.h");
        include!("table_shim.h");

        type OwnedColumn = crate::column::ffi::OwnedColumn;
        type OwnedTable = crate::table::ffi::OwnedTable;

        // ── Gather / Scatter ───────────────────────────────────────

        /// Gather rows from `table` according to `gather_map` (index column).
        /// `bounds_policy`: 0=DONT_CHECK, 1=NULLIFY.
        fn gather(
            table: &OwnedTable,
            gather_map: &OwnedColumn,
            bounds_policy: i32,
        ) -> Result<UniquePtr<OwnedTable>>;

        /// Scatter rows from `source` into `target` at positions in `scatter_map`.
        fn scatter(
            source: &OwnedTable,
            scatter_map: &OwnedColumn,
            target: &OwnedTable,
        ) -> Result<UniquePtr<OwnedTable>>;

        // ── Conditional Copy ───────────────────────────────────────

        /// Elementwise: select from `lhs` where mask is true, `rhs` where false.
        fn copy_if_else(
            lhs: &OwnedColumn,
            rhs: &OwnedColumn,
            boolean_mask: &OwnedColumn,
        ) -> Result<UniquePtr<OwnedColumn>>;

        // ── Slice / Split ──────────────────────────────────────────

        /// Extract a contiguous slice [begin, end) as an owned deep copy.
        fn slice_table(table: &OwnedTable, begin: i32, end: i32) -> Result<UniquePtr<OwnedTable>>;

        /// Opaque result of splitting a table into multiple parts.
        type SplitResult;

        /// Split a table at the given indices, returning all parts at once.
        fn split_table_all(table: &OwnedTable, splits: &[i32]) -> Result<UniquePtr<SplitResult>>;

        /// Return the number of parts in a split result.
        fn split_result_count(result: &SplitResult) -> i32;

        /// Move one part out of a split result by index.
        fn split_result_get(
            result: Pin<&mut SplitResult>,
            index: i32,
        ) -> Result<UniquePtr<OwnedTable>>;

        // ── Empty / Allocate ───────────────────────────────────────

        /// Create an empty column with the same type and size as `col`, all nulls.
        fn empty_like(col: &OwnedColumn) -> Result<UniquePtr<OwnedColumn>>;

        /// Create a column with the same type and size as `col`, with allocated
        /// but uninitialized data.
        /// `mask_policy`: 0=NEVER, 1=ALWAYS, 2=RETAIN.
        fn allocate_like(col: &OwnedColumn, mask_policy: i32) -> Result<UniquePtr<OwnedColumn>>;

        // ── In-place Copy ──────────────────────────────────────────

        /// Copy a range from `source` into `target` column (in-place).
        fn copy_range(
            source: &OwnedColumn,
            target: Pin<&mut OwnedColumn>,
            source_begin: i32,
            source_end: i32,
            target_begin: i32,
        ) -> Result<()>;
    }
}
