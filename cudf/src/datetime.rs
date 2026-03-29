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
}
