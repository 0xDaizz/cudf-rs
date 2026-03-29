//! Bridge definitions for libcudf list column operations.

#[cxx::bridge(namespace = "cudf_shims")]
pub mod ffi {
    unsafe extern "C++" {
        include!("lists/lists_shim.h");
        include!("column_shim.h");
        include!("table_shim.h");
        include!("scalar_shim.h");

        type OwnedColumn = crate::column::ffi::OwnedColumn;
        type OwnedTable = crate::table::ffi::OwnedTable;
        type OwnedScalar = crate::scalar::ffi::OwnedScalar;

        // ── Explode ────────────────────────────────────────────────

        /// Explode a list column, expanding each list element into its own row.
        fn lists_explode(table: &OwnedTable, explode_col_idx: i32)
        -> Result<UniquePtr<OwnedTable>>;

        /// Explode a list column, retaining null entries and empty lists.
        fn lists_explode_outer(
            table: &OwnedTable,
            explode_col_idx: i32,
        ) -> Result<UniquePtr<OwnedTable>>;

        // ── Sorting ───────────────────────────────────────────────

        /// Sort elements within each list row.
        /// `ascending`: true for ascending, false for descending.
        /// `null_order`: 0=BEFORE, 1=AFTER.
        fn lists_sort(
            col: &OwnedColumn,
            ascending: bool,
            null_order: i32,
        ) -> Result<UniquePtr<OwnedColumn>>;

        // ── Contains ──────────────────────────────────────────────

        /// Check whether each list row contains the given scalar value.
        fn lists_contains(
            col: &OwnedColumn,
            search_key: &OwnedScalar,
        ) -> Result<UniquePtr<OwnedColumn>>;

        /// Check whether each list row contains any null elements.
        fn lists_contains_nulls(col: &OwnedColumn) -> Result<UniquePtr<OwnedColumn>>;

        // ── Extract ───────────────────────────────────────────────

        /// Extract the element at `index` from each list row.
        fn lists_extract(col: &OwnedColumn, index: i32) -> Result<UniquePtr<OwnedColumn>>;
    }
}
