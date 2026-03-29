//! Bridge definitions for libcudf quantile operations.
//!
//! Provides GPU-accelerated quantile and percentile approximation functions
//! for columns and tables.

#[cxx::bridge(namespace = "cudf_shims")]
pub mod ffi {
    unsafe extern "C++" {
        include!("column_shim.h");
        include!("table_shim.h");
        include!("quantiles_shim.h");

        type OwnedColumn = crate::column::ffi::OwnedColumn;
        type OwnedTable = crate::table::ffi::OwnedTable;

        /// Compute quantile(s) of a column.
        /// interp: 0=linear, 1=lower, 2=higher, 3=midpoint, 4=nearest
        fn quantile(col: &OwnedColumn, q: &[f64], interp: i32) -> Result<UniquePtr<OwnedColumn>>;

        /// Compute quantiles of a table (row-wise).
        fn quantiles_table(
            table: &OwnedTable,
            q: &[f64],
            interp: i32,
            is_input_sorted: bool,
            orders: &[i32],
            null_orders: &[i32],
        ) -> Result<UniquePtr<OwnedTable>>;

        /// Compute percentile approximation using t-digest.
        fn percentile_approx(
            tdigest_col: &OwnedColumn,
            percentiles: &[f64],
        ) -> Result<UniquePtr<OwnedColumn>>;
    }
}
