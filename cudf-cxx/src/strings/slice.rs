//! Bridge definitions for libcudf string slice operations.

#[cxx::bridge(namespace = "cudf_shims")]
pub mod ffi {
    unsafe extern "C++" {
        include!("strings/slice_shim.h");
        include!("column_shim.h");

        type OwnedColumn = crate::column::ffi::OwnedColumn;

        /// Extract a substring from each string, from `start` to `stop` (exclusive).
        /// Negative indices are not supported; use 0 for start and -1 for stop
        /// to indicate end-of-string.
        fn str_slice(col: &OwnedColumn, start: i32, stop: i32) -> Result<UniquePtr<OwnedColumn>>;

        /// Extract substrings using per-row start/stop integer columns.
        fn str_slice_column(
            col: &OwnedColumn,
            starts: &OwnedColumn,
            stops: &OwnedColumn,
        ) -> Result<UniquePtr<OwnedColumn>>;
    }
}
