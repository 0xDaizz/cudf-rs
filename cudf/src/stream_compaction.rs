//! Stream compaction operations for tables and columns.
//!
//! Provides GPU-accelerated null dropping, boolean masking, and
//! duplicate removal operations.
//!
//! # Examples
//!
//! ```rust,no_run
//! use cudf::{Column, Table};
//! use cudf::stream_compaction::DuplicateKeepOption;
//!
//! let col = Column::from_slice(&[1i32, 2, 2, 3, 3, 3]).unwrap();
//! let table = Table::new(vec![col]).unwrap();
//! let unique_table = table.unique(&[0], DuplicateKeepOption::First).unwrap();
//! ```

use crate::column::Column;
use crate::error::{CudfError, Result};
use crate::table::Table;

/// Controls which duplicate row to keep.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DuplicateKeepOption {
    /// Keep the first occurrence of each duplicate.
    First = 0,
    /// Keep the last occurrence of each duplicate.
    Last = 1,
    /// Keep any single occurrence of each duplicate.
    Any = 2,
    /// Remove all duplicates entirely.
    None = 3,
}

impl DuplicateKeepOption {
    fn as_i32(self) -> i32 {
        self as i32
    }
}

impl Table {
    /// Drop rows where any of the specified key columns contain nulls.
    ///
    /// `key_columns` specifies which column indices to check for nulls.
    /// `threshold` is the minimum number of non-null values in key columns
    /// required to keep a row.
    ///
    /// # Errors
    ///
    /// Returns an error if any key column index is out of bounds.
    pub fn drop_nulls(&self, key_columns: &[usize], threshold: usize) -> Result<Table> {
        let keys: Vec<i32> = key_columns.iter().map(|&k| k as i32).collect();
        let raw = cudf_cxx::stream_compaction::ffi::drop_nulls_table(
            &self.inner,
            &keys,
            threshold as i32,
        )
        .map_err(CudfError::from_cxx)?;
        Ok(Table { inner: raw })
    }

    /// Keep only rows where the boolean mask column is `true`.
    ///
    /// The mask column must be of BOOL8 type and have the same number
    /// of rows as the table.
    ///
    /// # Errors
    ///
    /// Returns an error if the mask column type or length is invalid.
    pub fn apply_boolean_mask(&self, mask: &Column) -> Result<Table> {
        let raw = cudf_cxx::stream_compaction::ffi::apply_boolean_mask(
            &self.inner,
            &mask.inner,
        )
        .map_err(CudfError::from_cxx)?;
        Ok(Table { inner: raw })
    }

    /// Return a table with unique rows based on the specified key columns.
    ///
    /// The result is sorted in the same order as the input. `keep` controls
    /// which duplicate to retain.
    ///
    /// # Errors
    ///
    /// Returns an error if any key column index is out of bounds.
    pub fn unique(&self, key_columns: &[usize], keep: DuplicateKeepOption) -> Result<Table> {
        let keys: Vec<i32> = key_columns.iter().map(|&k| k as i32).collect();
        let raw = cudf_cxx::stream_compaction::ffi::unique(
            &self.inner,
            &keys,
            keep.as_i32(),
            0, // null_equality: EQUAL
        )
        .map_err(CudfError::from_cxx)?;
        Ok(Table { inner: raw })
    }

    /// Return a table with distinct rows based on the specified key columns.
    ///
    /// Unlike `unique`, `distinct` does not preserve the relative order of
    /// equivalent rows.
    ///
    /// # Errors
    ///
    /// Returns an error if any key column index is out of bounds.
    pub fn distinct(&self, key_columns: &[usize], keep: DuplicateKeepOption) -> Result<Table> {
        let keys: Vec<i32> = key_columns.iter().map(|&k| k as i32).collect();
        let raw = cudf_cxx::stream_compaction::ffi::distinct(
            &self.inner,
            &keys,
            keep.as_i32(),
            0, // null_equality: EQUAL
        )
        .map_err(CudfError::from_cxx)?;
        Ok(Table { inner: raw })
    }

    /// Drop rows where any of the specified key columns contain NaN.
    ///
    /// # Errors
    ///
    /// Returns an error if any key column index is out of bounds or
    /// if a key column is not a floating-point type.
    pub fn drop_nans(&self, key_columns: &[usize]) -> Result<Table> {
        let keys: Vec<i32> = key_columns.iter().map(|&k| k as i32).collect();
        let raw = cudf_cxx::stream_compaction::ffi::drop_nans(
            &self.inner,
            &keys,
        )
        .map_err(CudfError::from_cxx)?;
        Ok(Table { inner: raw })
    }
}

impl Column {
    /// Drop null values from this column, returning a new column.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use cudf::Column;
    ///
    /// let col = Column::from_slice(&[1i32, 2, 3]).unwrap();
    /// let no_nulls = col.drop_nulls().unwrap();
    /// ```
    pub fn drop_nulls(&self) -> Result<Column> {
        let raw = cudf_cxx::stream_compaction::ffi::drop_nulls_column(&self.inner)
            .map_err(CudfError::from_cxx)?;
        Ok(Column { inner: raw })
    }

    /// Count the number of distinct elements in this column.
    ///
    /// Null values are included in the count as a single distinct value.
    /// NaN values are treated as valid (not null).
    pub fn distinct_count(&self) -> Result<usize> {
        let count = cudf_cxx::stream_compaction::ffi::distinct_count_column(
            &self.inner,
            0, // null_handling: INCLUDE
            0, // nan_handling: NAN_IS_VALID
        )
        .map_err(CudfError::from_cxx)?;
        Ok(count as usize)
    }
}
