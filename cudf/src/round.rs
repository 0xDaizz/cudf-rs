//! GPU-accelerated rounding operations.
//!
//! Provides numeric rounding for [`Column`]s.
//!
//! # Examples
//!
//! ```rust,no_run
//! use cudf::Column;
//!
//! let col = Column::from_slice(&[1.2345f64, 2.6789, 3.1415]).unwrap();
//! let rounded = col.round(2).unwrap();
//! ```

use crate::column::Column;
use crate::error::{CudfError, Result};

impl Column {
    /// Round this numeric column to the specified number of decimal places.
    ///
    /// Uses HALF_UP rounding mode (standard mathematical rounding).
    pub fn round(&self, decimal_places: i32) -> Result<Column> {
        let raw = cudf_cxx::round::ffi::round_column(&self.inner, decimal_places)
            .map_err(CudfError::from_cxx)?;
        Ok(Column { inner: raw })
    }
}
