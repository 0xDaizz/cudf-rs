//! Bridge definitions for libcudf join operations.
//!
//! Join functions return gather maps (index columns) that can be used
//! to construct the joined table via `gather()`.

#[cxx::bridge(namespace = "cudf_shims")]
pub mod ffi {
    unsafe extern "C++" {
        include!("join_shim.h");
        include!("table_shim.h");

        type OwnedTable = crate::table::ffi::OwnedTable;

        /// Inner join: returns a 2-column table [left_indices, right_indices].
        fn inner_join(
            left_keys: &OwnedTable,
            right_keys: &OwnedTable,
        ) -> Result<UniquePtr<OwnedTable>>;

        /// Left join: returns a 2-column table [left_indices, right_indices].
        fn left_join(
            left_keys: &OwnedTable,
            right_keys: &OwnedTable,
        ) -> Result<UniquePtr<OwnedTable>>;

        /// Full outer join: returns a 2-column table [left_indices, right_indices].
        fn full_join(
            left_keys: &OwnedTable,
            right_keys: &OwnedTable,
        ) -> Result<UniquePtr<OwnedTable>>;

        /// Cross join: cartesian product of two tables.
        fn cross_join(
            left: &OwnedTable,
            right: &OwnedTable,
        ) -> Result<UniquePtr<OwnedTable>>;
    }
}
