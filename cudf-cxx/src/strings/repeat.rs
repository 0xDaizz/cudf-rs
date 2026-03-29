//! Bridge definitions for libcudf string repeat operations.

#[cxx::bridge(namespace = "cudf_shims")]
pub mod ffi {
    unsafe extern "C++" {
        include!("strings/repeat_shim.h");
        include!("column_shim.h");

        type OwnedColumn = crate::column::ffi::OwnedColumn;

        /// Repeat each string `count` times.
        fn str_repeat(col: &OwnedColumn, count: i32) -> Result<UniquePtr<OwnedColumn>>;

        /// Repeat each string by the count in a per-row column.
        fn str_repeat_per_row(
            col: &OwnedColumn,
            counts: &OwnedColumn,
        ) -> Result<UniquePtr<OwnedColumn>>;
    }
}
