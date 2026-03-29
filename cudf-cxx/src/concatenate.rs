//! Bridge definitions for libcudf concatenation operations.
//!
//! Uses a builder pattern to collect column/table views before
//! performing the concatenation, working around cxx's limitations
//! with slices of opaque types.

#[cxx::bridge(namespace = "cudf_shims")]
pub mod ffi {
    unsafe extern "C++" {
        include!("concatenate_shim.h");
        include!("column_shim.h");
        include!("table_shim.h");

        type OwnedColumn = crate::column::ffi::OwnedColumn;
        type OwnedTable = crate::table::ffi::OwnedTable;

        // ── Column Concatenation Builder ───────────────────────────

        /// Builder that accumulates column views for concatenation.
        type ColumnConcatBuilder;

        /// Add a column to the builder.
        fn add(self: Pin<&mut ColumnConcatBuilder>, col: &OwnedColumn);

        /// Concatenate all added columns into one.
        fn build(self: &ColumnConcatBuilder) -> Result<UniquePtr<OwnedColumn>>;

        /// Create a new ColumnConcatBuilder.
        fn new_column_concat_builder() -> UniquePtr<ColumnConcatBuilder>;

        // ── Table Concatenation Builder ────────────────────────────

        /// Builder that accumulates table views for concatenation.
        type TableConcatBuilder;

        /// Add a table to the builder.
        fn add(self: Pin<&mut TableConcatBuilder>, table: &OwnedTable);

        /// Concatenate all added tables into one.
        fn build(self: &TableConcatBuilder) -> Result<UniquePtr<OwnedTable>>;

        /// Create a new TableConcatBuilder.
        fn new_table_concat_builder() -> UniquePtr<TableConcatBuilder>;
    }
}
