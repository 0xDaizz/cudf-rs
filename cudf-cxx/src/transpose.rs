//! Bridge definitions for libcudf transpose operations.
//!
//! Provides GPU-accelerated table transposition.

#[cxx::bridge(namespace = "cudf_shims")]
pub mod ffi {
    unsafe extern "C++" {
        include!("transpose_shim.h");
        include!("table_shim.h");
        type OwnedTable = crate::table::ffi::OwnedTable;

        /// Transpose a table (swap rows and columns).
        /// All columns must have the same data type.
        fn transpose_table(table: &OwnedTable) -> Result<UniquePtr<OwnedTable>>;
    }
}
