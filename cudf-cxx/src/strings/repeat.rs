//! Bridge definitions for libcudf string repeat operations.

#[cxx::bridge(namespace = "cudf_shims")]
pub mod ffi {
    unsafe extern "C++" {
        include!("strings/repeat_shim.h");
        include!("column_shim.h");

        type OwnedColumn = crate::column::ffi::OwnedColumn;

        /// Repeat each string `count` times.
        fn str_repeat(col: &OwnedColumn, count: i32) -> Result<UniquePtr<OwnedColumn>>;
    }
}
