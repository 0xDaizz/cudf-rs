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

        // ── Explode Position ──────────────────────────────────────

        /// Explode a list column with position indices.
        fn lists_explode_position(
            table: &OwnedTable,
            explode_col_idx: i32,
        ) -> Result<UniquePtr<OwnedTable>>;

        /// Explode outer with position indices.
        fn lists_explode_outer_position(
            table: &OwnedTable,
            explode_col_idx: i32,
        ) -> Result<UniquePtr<OwnedTable>>;

        // ── Count Elements ────────────────────────────────────────

        /// Count elements in each list row.
        fn lists_count_elements(col: &OwnedColumn) -> Result<UniquePtr<OwnedColumn>>;

        // ── Index Of ──────────────────────────────────────────────

        /// Find position of scalar in each list row.
        fn lists_index_of_scalar(
            col: &OwnedColumn,
            key: &OwnedScalar,
        ) -> Result<UniquePtr<OwnedColumn>>;

        // ── Combine ───────────────────────────────────────────────

        /// Concatenate lists across columns (row-wise).
        fn lists_concatenate_rows(table: &OwnedTable) -> Result<UniquePtr<OwnedColumn>>;

        /// Concatenate nested list elements within each row.
        fn lists_concatenate_list_elements(col: &OwnedColumn) -> Result<UniquePtr<OwnedColumn>>;

        // ── Filling (sequences) ──────────────────────────────────

        /// Generate list column of arithmetic sequences.
        fn lists_sequences(
            starts: &OwnedColumn,
            sizes: &OwnedColumn,
        ) -> Result<UniquePtr<OwnedColumn>>;

        // ── Gather ────────────────────────────────────────────────

        /// Gather elements from lists based on per-row gather maps.
        fn lists_segmented_gather(
            col: &OwnedColumn,
            gather_map: &OwnedColumn,
        ) -> Result<UniquePtr<OwnedColumn>>;

        // ── Set Operations ────────────────────────────────────────

        /// Check if lists at each row overlap.
        fn lists_have_overlap(
            lhs: &OwnedColumn,
            rhs: &OwnedColumn,
        ) -> Result<UniquePtr<OwnedColumn>>;

        /// Distinct elements common to both lists.
        fn lists_intersect_distinct(
            lhs: &OwnedColumn,
            rhs: &OwnedColumn,
        ) -> Result<UniquePtr<OwnedColumn>>;

        /// Distinct elements found in either list.
        fn lists_union_distinct(
            lhs: &OwnedColumn,
            rhs: &OwnedColumn,
        ) -> Result<UniquePtr<OwnedColumn>>;

        /// Distinct elements in lhs but not rhs.
        fn lists_difference_distinct(
            lhs: &OwnedColumn,
            rhs: &OwnedColumn,
        ) -> Result<UniquePtr<OwnedColumn>>;

        // ── Reverse ───────────────────────────────────────────────

        /// Reverse elements within each list row.
        fn lists_reverse(col: &OwnedColumn) -> Result<UniquePtr<OwnedColumn>>;

        // ── Stream Compaction ─────────────────────────────────────

        /// Filter list elements using a boolean mask list column.
        fn lists_apply_boolean_mask(
            col: &OwnedColumn,
            mask: &OwnedColumn,
        ) -> Result<UniquePtr<OwnedColumn>>;

        /// Remove duplicate elements within each list row.
        fn lists_distinct(col: &OwnedColumn) -> Result<UniquePtr<OwnedColumn>>;

        // ── New Low Priority ─────────────────────────────────────

        /// Stable sort elements within each list row.
        fn lists_stable_sort(
            col: &OwnedColumn,
            ascending: bool,
            null_order: i32,
        ) -> Result<UniquePtr<OwnedColumn>>;

        /// Extract elements using per-row indices from a column.
        fn lists_extract_column_index(
            col: &OwnedColumn,
            indices: &OwnedColumn,
        ) -> Result<UniquePtr<OwnedColumn>>;

        /// Check whether each list row contains the corresponding value from search_keys column.
        fn lists_contains_column(
            col: &OwnedColumn,
            search_keys: &OwnedColumn,
        ) -> Result<UniquePtr<OwnedColumn>>;
    }
}
