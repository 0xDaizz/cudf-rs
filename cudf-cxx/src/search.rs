//! Bridge definitions for libcudf search operations.
//!
//! Provides GPU-accelerated binary search and containment checks.

#[cxx::bridge(namespace = "cudf_shims")]
pub mod ffi {
    unsafe extern "C++" {
        include!("search_shim.h");
        include!("table_shim.h");
        include!("column_shim.h");
        type OwnedTable = crate::table::ffi::OwnedTable;
        type OwnedColumn = crate::column::ffi::OwnedColumn;

        /// Find the lower bound indices for each row in `values` within a sorted `table`.
        fn lower_bound(
            table: &OwnedTable,
            values: &OwnedTable,
            orders: &[i32],
            null_orders: &[i32],
        ) -> Result<UniquePtr<OwnedColumn>>;

        /// Find the upper bound indices for each row in `values` within a sorted `table`.
        fn upper_bound(
            table: &OwnedTable,
            values: &OwnedTable,
            orders: &[i32],
            null_orders: &[i32],
        ) -> Result<UniquePtr<OwnedColumn>>;

        /// For each element in `needles`, check if it exists in `haystack`.
        fn contains_column(
            haystack: &OwnedColumn,
            needles: &OwnedColumn,
        ) -> Result<UniquePtr<OwnedColumn>>;
    }
}
