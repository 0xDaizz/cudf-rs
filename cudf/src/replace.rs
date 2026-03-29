//! GPU-accelerated null/NaN replacement and value clamping.
//!
//! Provides operations to replace null values, replace NaN values, and
//! clamp column values to a specified range.
//!
//! # Examples
//!
//! ```rust,no_run
//! use cudf::{Column, Scalar};
//!
//! let col = Column::from_slice(&[1.0f64, f64::NAN, 3.0]).unwrap();
//! let replacement = Scalar::new(0.0f64).unwrap();
//! let clean = col.replace_nans_scalar(&replacement).unwrap();
//! ```

use crate::column::Column;
use crate::error::{CudfError, Result};
use crate::scalar::Scalar;

impl Column {
    /// Replace null values with corresponding values from `replacement`.
    ///
    /// `self` and `replacement` must have the same type and size.
    pub fn replace_nulls_column(&self, replacement: &Column) -> Result<Column> {
        let raw = cudf_cxx::replace::ffi::replace_nulls_column(&self.inner, &replacement.inner)
            .map_err(CudfError::from_cxx)?;
        Ok(Column { inner: raw })
    }

    /// Replace null values with a scalar.
    ///
    /// `self` and `replacement` must have the same type.
    pub fn replace_nulls_scalar(&self, replacement: &Scalar) -> Result<Column> {
        let raw = cudf_cxx::replace::ffi::replace_nulls_scalar(&self.inner, &replacement.inner)
            .map_err(CudfError::from_cxx)?;
        Ok(Column { inner: raw })
    }

    /// Replace NaN values with a scalar.
    ///
    /// Only valid for floating-point columns.
    pub fn replace_nans_scalar(&self, replacement: &Scalar) -> Result<Column> {
        let raw = cudf_cxx::replace::ffi::replace_nans_scalar(&self.inner, &replacement.inner)
            .map_err(CudfError::from_cxx)?;
        Ok(Column { inner: raw })
    }

    /// Replace NaN values with corresponding values from `replacement`.
    ///
    /// Both columns must be floating-point and the same size.
    pub fn replace_nans_column(&self, replacement: &Column) -> Result<Column> {
        let raw = cudf_cxx::replace::ffi::replace_nans_column(&self.inner, &replacement.inner)
            .map_err(CudfError::from_cxx)?;
        Ok(Column { inner: raw })
    }

    /// Clamp values to the range [lo, hi].
    ///
    /// Values below `lo` become `lo`, values above `hi` become `hi`.
    /// If `lo` is null/invalid it is treated as the type minimum.
    /// If `hi` is null/invalid it is treated as the type maximum.
    pub fn clamp(&self, lo: &Scalar, hi: &Scalar) -> Result<Column> {
        let raw = cudf_cxx::replace::ffi::clamp(&self.inner, &lo.inner, &hi.inner)
            .map_err(CudfError::from_cxx)?;
        Ok(Column { inner: raw })
    }

    /// Normalize -NaN to +NaN and -0.0 to +0.0.
    ///
    /// Only valid for floating-point columns. Useful before comparison
    /// or hashing operations.
    pub fn normalize_nans_and_zeros(&self) -> Result<Column> {
        let raw = cudf_cxx::replace::ffi::normalize_nans_and_zeros(&self.inner)
            .map_err(CudfError::from_cxx)?;
        Ok(Column { inner: raw })
    }
}
