//! Bridge definitions for libcudf transform operations.
//!
//! Provides GPU-accelerated data transformations such as NaN-to-null
//! conversion and boolean mask generation.

#[cxx::bridge(namespace = "cudf_shims")]
pub mod ffi {
    unsafe extern "C++" {
        include!("transform_shim.h");
        include!("column_shim.h");
        include!("table_shim.h");
        type OwnedColumn = crate::column::ffi::OwnedColumn;
        type OwnedTable = crate::table::ffi::OwnedTable;

        /// Replace NaN values with nulls in a floating-point column.
        fn nans_to_nulls(col: &OwnedColumn) -> Result<UniquePtr<OwnedColumn>>;

        /// Convert a boolean column to a bitmask (host bytes).
        fn bools_to_mask(col: &OwnedColumn) -> Result<Vec<u8>>;

        /// Factorize a table: returns the distinct keys table.
        /// The encoded indices column is written to `out_indices`.
        fn encode_table(
            input: &OwnedTable,
            out_indices: &mut UniquePtr<OwnedColumn>,
        ) -> Result<UniquePtr<OwnedTable>>;

        /// One-hot-encode a column against a set of categories.
        fn one_hot_encode(
            input: &OwnedColumn,
            categories: &OwnedColumn,
        ) -> Result<UniquePtr<OwnedTable>>;

        /// Convert a bitmask (host bytes) to a boolean column.
        fn mask_to_bools(
            mask_data: &[u8],
            begin_bit: i32,
            end_bit: i32,
        ) -> Result<UniquePtr<OwnedColumn>>;

        /// Compute per-row bit count for a table.
        fn row_bit_count(table: &OwnedTable) -> Result<UniquePtr<OwnedColumn>>;
    }
}
