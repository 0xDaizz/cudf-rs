//! GPU-accelerated copying operations.
//!
//! Provides gather, scatter, slice, split, and conditional copy operations
//! for [`Table`]s and [`Column`]s.
//!
//! # Examples
//!
//! ```rust,no_run
//! use cudf::{Column, Table};
//!
//! let col = Column::from_slice(&[10i32, 20, 30, 40, 50]).unwrap();
//! let table = Table::new(vec![col]).unwrap();
//!
//! // Gather rows 0, 2, 4
//! let indices = Column::from_slice(&[0i32, 2, 4]).unwrap();
//! let gathered = table.gather(&indices).unwrap();
//! assert_eq!(gathered.num_rows(), 3);
//! ```

use crate::column::Column;
use crate::error::{CudfError, Result};
use crate::table::Table;

/// Policy for out-of-bounds indices in gather operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutOfBoundsPolicy {
    /// Do not check bounds (undefined behavior if out of range).
    DontCheck = 0,
    /// Replace out-of-bounds values with null.
    Nullify = 1,
}

// -- Table methods --

impl Table {
    /// Gather rows from this table using an index column.
    ///
    /// The `gather_map` column contains integer indices specifying which
    /// rows to select. Out-of-bounds indices produce null values.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use cudf::{Column, Table};
    ///
    /// let col = Column::from_slice(&[10i32, 20, 30]).unwrap();
    /// let table = Table::new(vec![col]).unwrap();
    /// let map = Column::from_slice(&[2i32, 0, 1]).unwrap();
    /// let result = table.gather(&map).unwrap();
    /// ```
    pub fn gather(&self, gather_map: &Column) -> Result<Table> {
        let raw = cudf_cxx::copying::ffi::gather(
            &self.inner,
            &gather_map.inner,
            OutOfBoundsPolicy::Nullify as i32,
        )
        .map_err(CudfError::from_cxx)?;

        Ok(Table { inner: raw })
    }

    /// Scatter rows from this table into `target` at positions in `scatter_map`.
    ///
    /// For each row `i` in `self`, the row is placed at position
    /// `scatter_map[i]` in the result (which starts as a copy of `target`).
    pub fn scatter(&self, scatter_map: &Column, target: &Table) -> Result<Table> {
        let raw = cudf_cxx::copying::ffi::scatter(&self.inner, &scatter_map.inner, &target.inner)
            .map_err(CudfError::from_cxx)?;

        Ok(Table { inner: raw })
    }

    /// Extract a contiguous slice `[begin, end)` as an owned table.
    ///
    /// This creates a deep copy of the sliced data.
    ///
    /// # Errors
    ///
    /// Returns an error if `begin > end` or `end > num_rows()`.
    pub fn slice(&self, begin: usize, end: usize) -> Result<Table> {
        if begin > end {
            return Err(CudfError::InvalidArgument(format!(
                "slice begin ({}) must not exceed end ({})",
                begin, end
            )));
        }
        if end > self.num_rows() {
            return Err(CudfError::IndexOutOfBounds {
                index: end,
                size: self.num_rows(),
            });
        }

        let raw = cudf_cxx::copying::ffi::slice_table(&self.inner, begin as i32, end as i32)
            .map_err(CudfError::from_cxx)?;

        Ok(Table { inner: raw })
    }

    /// Split this table at the given row indices, returning owned copies.
    ///
    /// For indices `[a, b]`, returns 3 tables: `[0, a)`, `[a, b)`, `[b, num_rows)`.
    ///
    /// # Errors
    ///
    /// Returns an error if indices are not strictly increasing or out of bounds.
    pub fn split(&self, indices: &[usize]) -> Result<Vec<Table>> {
        let idx: Vec<i32> = indices.iter().map(|&i| i as i32).collect();
        let mut result = cudf_cxx::copying::ffi::split_table_all(&self.inner, &idx)
            .map_err(CudfError::from_cxx)?;
        let count = cudf_cxx::copying::ffi::split_result_count(&result);
        let mut tables = Vec::with_capacity(count as usize);
        for i in 0..count {
            let raw = cudf_cxx::copying::ffi::split_result_get(result.pin_mut(), i)
                .map_err(CudfError::from_cxx)?;
            tables.push(Table { inner: raw });
        }
        Ok(tables)
    }
}

// -- Column methods --

impl Column {
    /// Select elements from two columns based on a boolean mask.
    ///
    /// For each element: if `mask[i]` is true, take from `self`;
    /// if false, take from `other`.
    pub fn copy_if_else(&self, other: &Column, mask: &Column) -> Result<Column> {
        let raw = cudf_cxx::copying::ffi::copy_if_else(&self.inner, &other.inner, &mask.inner)
            .map_err(CudfError::from_cxx)?;

        Ok(Column { inner: raw })
    }

    /// Create an empty column with the same type and size, all nulls.
    pub fn empty_like(&self) -> Result<Column> {
        let raw = cudf_cxx::copying::ffi::empty_like(&self.inner).map_err(CudfError::from_cxx)?;

        Ok(Column { inner: raw })
    }
}
