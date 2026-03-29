//! Bridge definitions for libcudf table types.
//!
//! A `Table` is an ordered collection of `Column`s, analogous to a DataFrame.

#[cxx::bridge(namespace = "cudf_shims")]
pub mod ffi {
    unsafe extern "C++" {
        include!("table_shim.h");
        include!("column_shim.h");

        type OwnedColumn = crate::column::ffi::OwnedColumn;

        /// Opaque owning handle wrapping `std::unique_ptr<cudf::table>`.
        type OwnedTable;

        /// Builder for constructing a table column-by-column (avoids Vec<UniquePtr<T>>).
        type TableBuilder;

        // ── Accessors ──────────────────────────────────────────────

        /// Number of columns in this table.
        fn num_columns(self: &OwnedTable) -> i32;

        /// Number of rows in this table.
        fn num_rows(self: &OwnedTable) -> i32;

        // ── Builder pattern for table construction ─────────────────

        /// Create a new table builder.
        fn table_builder_new() -> UniquePtr<TableBuilder>;

        /// Add a column to the builder.
        fn add_column(self: Pin<&mut TableBuilder>, col: UniquePtr<OwnedColumn>);

        /// Build the table from the accumulated columns.
        fn build(self: Pin<&mut TableBuilder>) -> Result<UniquePtr<OwnedTable>>;

        // ── Column access ──────────────────────────────────────────

        /// Extract a column from the table by index (deep copy).
        fn table_get_column(table: &OwnedTable, index: i32) -> Result<UniquePtr<OwnedColumn>>;

        /// Release a single column by index (takes ownership, invalidates that slot).
        /// Use reverse order to avoid index shifting.
        fn table_release_column(
            table: Pin<&mut OwnedTable>,
            index: i32,
        ) -> Result<UniquePtr<OwnedColumn>>;

        // ── TableWithMetadata (IO readers with column names) ──────

        /// Opaque owning handle wrapping table + column name metadata.
        type OwnedTableWithMetadata;

        /// Number of columns in the metadata wrapper.
        fn table_meta_num_columns(meta: &OwnedTableWithMetadata) -> i32;

        /// Get the column name at a given index.
        fn table_meta_column_name(meta: &OwnedTableWithMetadata, index: i32) -> Result<String>;

        /// Extract the inner table, consuming the metadata wrapper.
        fn table_meta_into_table(
            meta: UniquePtr<OwnedTableWithMetadata>,
        ) -> Result<UniquePtr<OwnedTable>>;
    }
}
