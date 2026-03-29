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

    /// Replace null values using a fill policy.
    ///
    /// `Preceding` fills each null with the first non-null value before it.
    /// `Following` fills each null with the first non-null value after it.
    pub fn replace_nulls_policy(&self, policy: NullReplacePolicy) -> Result<Column> {
        let raw = cudf_cxx::replace::ffi::replace_nulls_policy(&self.inner, policy as i32)
            .map_err(CudfError::from_cxx)?;
        Ok(Column { inner: raw })
    }

    /// Find and replace all occurrences of specified values.
    ///
    /// For each element in `self`, if it matches a value in `old_values`,
    /// it is replaced with the corresponding value in `new_values`.
    /// `old_values` and `new_values` must have the same length and type
    /// as `self`.
    pub fn find_and_replace_all(&self, old_values: &Column, new_values: &Column) -> Result<Column> {
        let raw = cudf_cxx::replace::ffi::find_and_replace_all(
            &self.inner,
            &old_values.inner,
            &new_values.inner,
        )
        .map_err(CudfError::from_cxx)?;
        Ok(Column { inner: raw })
    }
}

impl Column {
    /// Normalize -NaN to +NaN and -0.0 to +0.0 in-place.
    ///
    /// Modifies the column directly without creating a copy.
    /// Only valid for floating-point columns.
    pub fn normalize_nans_and_zeros_inplace(&mut self) -> Result<()> {
        cudf_cxx::replace::ffi::normalize_nans_and_zeros_inplace(self.inner.pin_mut())
            .map_err(CudfError::from_cxx)
    }

    /// Clamp values with custom replacement values.
    ///
    /// Values below `lo` are replaced with `lo_replace`.
    /// Values above `hi` are replaced with `hi_replace`.
    pub fn clamp_with_replace(
        &self,
        lo: &Scalar,
        lo_replace: &Scalar,
        hi: &Scalar,
        hi_replace: &Scalar,
    ) -> Result<Column> {
        let raw = cudf_cxx::replace::ffi::clamp_with_replace(
            &self.inner,
            &lo.inner,
            &lo_replace.inner,
            &hi.inner,
            &hi_replace.inner,
        )
        .map_err(CudfError::from_cxx)?;
        Ok(Column { inner: raw })
    }
}

/// Policy for replacing null values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NullReplacePolicy {
    /// Replace with the first non-null value preceding the null.
    Preceding = 0,
    /// Replace with the first non-null value following the null.
    Following = 1,
}
