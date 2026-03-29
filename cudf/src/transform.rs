//! GPU-accelerated data transformation operations.
//!
//! Provides NaN-to-null conversion and boolean mask generation for [`Column`]s.
//!
//! # Examples
//!
//! ```rust,no_run
//! use cudf::Column;
//!
//! let col = Column::from_slice(&[1.0f64, f64::NAN, 3.0]).unwrap();
//! let cleaned = col.nans_to_nulls().unwrap();
//! assert!(cleaned.has_nulls());
//! ```

use crate::column::Column;
use crate::error::{CudfError, Result};

impl Column {
    /// Replace NaN values with nulls in a floating-point column.
    ///
    /// Returns a new column with the same data but NaN values replaced by nulls.
    pub fn nans_to_nulls(&self) -> Result<Column> {
        let raw = cudf_cxx::transform::ffi::nans_to_nulls(&self.inner)
            .map_err(CudfError::from_cxx)?;
        Ok(Column { inner: raw })
    }

    /// Convert a boolean column to a bitmask (host bytes).
    ///
    /// Each bit in the output corresponds to one element:
    /// bit is 1 if the boolean value is true, 0 if false.
    pub fn bools_to_mask(&self) -> Result<Vec<u8>> {
        cudf_cxx::transform::ffi::bools_to_mask(&self.inner)
            .map_err(CudfError::from_cxx)
    }
}
