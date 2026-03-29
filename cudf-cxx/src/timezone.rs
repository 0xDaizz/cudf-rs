//! Bridge definitions for libcudf timezone operations.

#[cxx::bridge(namespace = "cudf_shims")]
pub mod ffi {
    unsafe extern "C++" {
        include!("timezone_shim.h");
        include!("table_shim.h");

        type OwnedTable = crate::table::ffi::OwnedTable;

        /// Create a timezone transition table for converting ORC timestamps to UTC.
        fn make_timezone_transition_table(timezone_name: &str) -> Result<UniquePtr<OwnedTable>>;
    }
}
