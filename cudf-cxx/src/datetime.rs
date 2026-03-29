//! Bridge definitions for libcudf datetime extraction operations.
//!
//! Provides GPU-accelerated extraction of date/time components from
//! timestamp columns.

#[cxx::bridge(namespace = "cudf_shims")]
pub mod ffi {
    unsafe extern "C++" {
        include!("datetime_shim.h");
        include!("column_shim.h");
        type OwnedColumn = crate::column::ffi::OwnedColumn;

        /// Extract year component from a timestamp column.
        fn extract_year(col: &OwnedColumn) -> Result<UniquePtr<OwnedColumn>>;

        /// Extract month component (1-12) from a timestamp column.
        fn extract_month(col: &OwnedColumn) -> Result<UniquePtr<OwnedColumn>>;

        /// Extract day component (1-31) from a timestamp column.
        fn extract_day(col: &OwnedColumn) -> Result<UniquePtr<OwnedColumn>>;

        /// Extract hour component (0-23) from a timestamp column.
        fn extract_hour(col: &OwnedColumn) -> Result<UniquePtr<OwnedColumn>>;

        /// Extract minute component (0-59) from a timestamp column.
        fn extract_minute(col: &OwnedColumn) -> Result<UniquePtr<OwnedColumn>>;

        /// Extract second component (0-59) from a timestamp column.
        fn extract_second(col: &OwnedColumn) -> Result<UniquePtr<OwnedColumn>>;

        /// Extract weekday (0=Monday, 6=Sunday) from a timestamp column.
        fn extract_weekday(col: &OwnedColumn) -> Result<UniquePtr<OwnedColumn>>;

        /// Extract day-of-year (1-366) from a timestamp column.
        fn extract_day_of_year(col: &OwnedColumn) -> Result<UniquePtr<OwnedColumn>>;
    }
}
