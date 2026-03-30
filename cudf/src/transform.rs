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
use crate::table::Table;
use crate::types::checked_i32;

impl Column {
    /// Replace NaN values with nulls in a floating-point column.
    ///
    /// Returns a new column with the same data but NaN values replaced by nulls.
    pub fn nans_to_nulls(&self) -> Result<Column> {
        let raw =
            cudf_cxx::transform::ffi::nans_to_nulls(&self.inner).map_err(CudfError::from_cxx)?;
        Ok(Column { inner: raw })
    }

    /// Convert a boolean column to a bitmask (host bytes).
    ///
    /// Each bit in the output corresponds to one element:
    /// bit is 1 if the boolean value is true, 0 if false.
    pub fn bools_to_mask(&self) -> Result<Vec<u8>> {
        cudf_cxx::transform::ffi::bools_to_mask(&self.inner).map_err(CudfError::from_cxx)
    }

    /// One-hot-encode this column against the given categories.
    ///
    /// Returns a table where each column corresponds to a category,
    /// containing boolean values indicating whether the input element
    /// matches that category.
    ///
    /// # Errors
    ///
    /// Returns an error if `self` and `categories` have different types.
    pub fn one_hot_encode(&self, categories: &Column) -> Result<Table> {
        let raw = cudf_cxx::transform::ffi::one_hot_encode(&self.inner, &categories.inner)
            .map_err(CudfError::from_cxx)?;
        Ok(Table { inner: raw })
    }
}

impl Table {
    /// Factorize this table (encode).
    ///
    /// Returns a tuple of (keys_table, indices_column) where `keys_table`
    /// contains the distinct rows in sorted order, and `indices_column`
    /// maps each input row to its corresponding key row index.
    pub fn encode(&self) -> Result<(Table, Column)> {
        let mut out_indices = cxx::UniquePtr::null();
        let keys_table = cudf_cxx::transform::ffi::encode_table(&self.inner, &mut out_indices)
            .map_err(CudfError::from_cxx)?;
        Ok((Table { inner: keys_table }, Column { inner: out_indices }))
    }

    /// Compute per-row bit count across all columns.
    ///
    /// Returns an `i32` column where each element is the approximate
    /// total number of bits used by all columns in that row.
    pub fn row_bit_count(&self) -> Result<Column> {
        let raw =
            cudf_cxx::transform::ffi::row_bit_count(&self.inner).map_err(CudfError::from_cxx)?;
        Ok(Column { inner: raw })
    }
}

/// Convert a bitmask (host bytes) to a boolean column.
///
/// Returns a `bool` column for each bit in `[begin_bit, end_bit)`.
/// Bit `i` set (1) produces `true`, unset (0) produces `false`.
pub fn mask_to_bools(mask_data: &[u8], begin_bit: usize, end_bit: usize) -> Result<Column> {
    let raw = cudf_cxx::transform::ffi::mask_to_bools(
        mask_data,
        checked_i32(begin_bit)?,
        checked_i32(end_bit)?,
    )
    .map_err(CudfError::from_cxx)?;
    Ok(Column { inner: raw })
}
