//! Bridge definitions for libcudf merge operations.
//!
//! Provides GPU-accelerated merging of pre-sorted tables.

#[cxx::bridge(namespace = "cudf_shims")]
pub mod ffi {
    unsafe extern "C++" {
        include!("merge_shim.h");
        include!("table_shim.h");
        type OwnedTable = crate::table::ffi::OwnedTable;

        /// Merge two pre-sorted tables into a single sorted table.
        /// `key_cols`: column indices to merge on.
        /// `orders`: 0=ascending, 1=descending (one per key column).
        /// `null_orders`: 0=before, 1=after (one per key column).
        fn merge_tables(
            left: &OwnedTable,
            right: &OwnedTable,
            key_cols: &[i32],
            orders: &[i32],
            null_orders: &[i32],
        ) -> Result<UniquePtr<OwnedTable>>;
    }
}
