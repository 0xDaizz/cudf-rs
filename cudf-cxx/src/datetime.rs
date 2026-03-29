//! Bridge definitions for libcudf datetime extraction operations.
//!
//! Provides GPU-accelerated extraction of date/time components from
//! timestamp columns.

#[cxx::bridge(namespace = "cudf_shims")]
pub mod ffi {
    unsafe extern "C++" {
        include!("datetime_shim.h");
        include!("column_shim.h");
        include!("scalar_shim.h");
        type OwnedColumn = crate::column::ffi::OwnedColumn;
        type OwnedScalar = crate::scalar::ffi::OwnedScalar;

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

        /// Get last day of the month for each timestamp.
        fn last_day_of_month(col: &OwnedColumn) -> Result<UniquePtr<OwnedColumn>>;

        /// Add months (scalar) to each timestamp.
        fn add_calendrical_months_scalar(
            col: &OwnedColumn,
            months: &OwnedScalar,
        ) -> Result<UniquePtr<OwnedColumn>>;

        /// Add months (column) to each timestamp.
        fn add_calendrical_months_column(
            col: &OwnedColumn,
            months: &OwnedColumn,
        ) -> Result<UniquePtr<OwnedColumn>>;

        /// Check if the year is a leap year.
        fn is_leap_year(col: &OwnedColumn) -> Result<UniquePtr<OwnedColumn>>;

        /// Get the number of days in the month.
        fn days_in_month(col: &OwnedColumn) -> Result<UniquePtr<OwnedColumn>>;

        /// Extract the quarter (1-4).
        fn extract_quarter(col: &OwnedColumn) -> Result<UniquePtr<OwnedColumn>>;

        /// Ceil datetimes to the given frequency.
        fn ceil_datetimes(col: &OwnedColumn, freq: i32) -> Result<UniquePtr<OwnedColumn>>;

        /// Floor datetimes to the given frequency.
        fn floor_datetimes(col: &OwnedColumn, freq: i32) -> Result<UniquePtr<OwnedColumn>>;

        /// Round datetimes to the nearest frequency.
        fn round_datetimes(col: &OwnedColumn, freq: i32) -> Result<UniquePtr<OwnedColumn>>;
    }
}
