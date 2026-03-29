//! GPU-accelerated datetime extraction operations.
//!
//! Provides extraction of date/time components from timestamp [`Column`]s.
//!
//! # Examples
//!
//! ```rust,no_run
//! use cudf::Column;
//!
//! // Assuming `ts_col` is a timestamp column:
//! // let years = ts_col.extract_year().unwrap();
//! ```

use crate::column::Column;
use crate::error::{CudfError, Result};
use crate::scalar::Scalar;

impl Column {
    /// Extract year component from a timestamp column.
    pub fn extract_year(&self) -> Result<Column> {
        let raw =
            cudf_cxx::datetime::ffi::extract_year(&self.inner).map_err(CudfError::from_cxx)?;
        Ok(Column { inner: raw })
    }

    /// Extract month component (1-12) from a timestamp column.
    pub fn extract_month(&self) -> Result<Column> {
        let raw =
            cudf_cxx::datetime::ffi::extract_month(&self.inner).map_err(CudfError::from_cxx)?;
        Ok(Column { inner: raw })
    }

    /// Extract day component (1-31) from a timestamp column.
    pub fn extract_day(&self) -> Result<Column> {
        let raw = cudf_cxx::datetime::ffi::extract_day(&self.inner).map_err(CudfError::from_cxx)?;
        Ok(Column { inner: raw })
    }

    /// Extract hour component (0-23) from a timestamp column.
    pub fn extract_hour(&self) -> Result<Column> {
        let raw =
            cudf_cxx::datetime::ffi::extract_hour(&self.inner).map_err(CudfError::from_cxx)?;
        Ok(Column { inner: raw })
    }

    /// Extract minute component (0-59) from a timestamp column.
    pub fn extract_minute(&self) -> Result<Column> {
        let raw =
            cudf_cxx::datetime::ffi::extract_minute(&self.inner).map_err(CudfError::from_cxx)?;
        Ok(Column { inner: raw })
    }

    /// Extract second component (0-59) from a timestamp column.
    pub fn extract_second(&self) -> Result<Column> {
        let raw =
            cudf_cxx::datetime::ffi::extract_second(&self.inner).map_err(CudfError::from_cxx)?;
        Ok(Column { inner: raw })
    }

    /// Extract weekday (0=Monday, 6=Sunday) from a timestamp column.
    pub fn extract_weekday(&self) -> Result<Column> {
        let raw =
            cudf_cxx::datetime::ffi::extract_weekday(&self.inner).map_err(CudfError::from_cxx)?;
        Ok(Column { inner: raw })
    }

    /// Extract day-of-year (1-366) from a timestamp column.
    pub fn extract_day_of_year(&self) -> Result<Column> {
        let raw = cudf_cxx::datetime::ffi::extract_day_of_year(&self.inner)
            .map_err(CudfError::from_cxx)?;
        Ok(Column { inner: raw })
    }

    /// Get the last day of the month for each timestamp.
    pub fn last_day_of_month(&self) -> Result<Column> {
        let raw =
            cudf_cxx::datetime::ffi::last_day_of_month(&self.inner).map_err(CudfError::from_cxx)?;
        Ok(Column { inner: raw })
    }

    /// Add months (scalar) to each timestamp.
    pub fn add_calendrical_months_scalar(&self, months: &Scalar) -> Result<Column> {
        let raw =
            cudf_cxx::datetime::ffi::add_calendrical_months_scalar(&self.inner, &months.inner)
                .map_err(CudfError::from_cxx)?;
        Ok(Column { inner: raw })
    }

    /// Add months (per-row column) to each timestamp.
    pub fn add_calendrical_months_column(&self, months: &Column) -> Result<Column> {
        let raw =
            cudf_cxx::datetime::ffi::add_calendrical_months_column(&self.inner, &months.inner)
                .map_err(CudfError::from_cxx)?;
        Ok(Column { inner: raw })
    }

    /// Check if the year of each date is a leap year.
    pub fn is_leap_year(&self) -> Result<Column> {
        let raw =
            cudf_cxx::datetime::ffi::is_leap_year(&self.inner).map_err(CudfError::from_cxx)?;
        Ok(Column { inner: raw })
    }

    /// Get the number of days in the month for each date.
    pub fn days_in_month(&self) -> Result<Column> {
        let raw =
            cudf_cxx::datetime::ffi::days_in_month(&self.inner).map_err(CudfError::from_cxx)?;
        Ok(Column { inner: raw })
    }

    /// Extract the quarter (1-4) from each timestamp.
    pub fn extract_quarter(&self) -> Result<Column> {
        let raw =
            cudf_cxx::datetime::ffi::extract_quarter(&self.inner).map_err(CudfError::from_cxx)?;
        Ok(Column { inner: raw })
    }

    /// Ceil datetimes to the given frequency.
    ///
    /// `freq`: 0=DAY, 1=HOUR, 2=MINUTE, 3=SECOND, 4=MILLISECOND, 5=MICROSECOND, 6=NANOSECOND.
    pub fn ceil_datetimes(&self, freq: i32) -> Result<Column> {
        let raw = cudf_cxx::datetime::ffi::ceil_datetimes(&self.inner, freq)
            .map_err(CudfError::from_cxx)?;
        Ok(Column { inner: raw })
    }

    /// Floor datetimes to the given frequency.
    ///
    /// `freq`: 0=DAY, 1=HOUR, 2=MINUTE, 3=SECOND, 4=MILLISECOND, 5=MICROSECOND, 6=NANOSECOND.
    pub fn floor_datetimes(&self, freq: i32) -> Result<Column> {
        let raw = cudf_cxx::datetime::ffi::floor_datetimes(&self.inner, freq)
            .map_err(CudfError::from_cxx)?;
        Ok(Column { inner: raw })
    }

    /// Round datetimes to the nearest multiple of the given frequency.
    ///
    /// `freq`: 0=DAY, 1=HOUR, 2=MINUTE, 3=SECOND, 4=MILLISECOND, 5=MICROSECOND, 6=NANOSECOND.
    pub fn round_datetimes(&self, freq: i32) -> Result<Column> {
        let raw = cudf_cxx::datetime::ffi::round_datetimes(&self.inner, freq)
            .map_err(CudfError::from_cxx)?;
        Ok(Column { inner: raw })
    }
}
