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
use crate::scalar::Scalar;
use crate::table::Table;
use crate::types::checked_i32;

/// Select elements from a scalar (true in mask) or a column (false in mask).
///
/// For each element: if `mask[i]` is true, take from `scalar`;
/// if false, take from `rhs`.
pub fn copy_if_else_scalar_col(scalar: &Scalar, rhs: &Column, mask: &Column) -> Result<Column> {
    let raw =
        cudf_cxx::copying::ffi::copy_if_else_scalar_col(&scalar.inner, &rhs.inner, &mask.inner)
            .map_err(CudfError::from_cxx)?;
    Ok(Column { inner: raw })
}

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

        let raw = cudf_cxx::copying::ffi::slice_table(
            &self.inner,
            checked_i32(begin)?,
            checked_i32(end)?,
        )
        .map_err(CudfError::from_cxx)?;

        Ok(Table { inner: raw })
    }

    /// Reverse the rows of this table.
    ///
    /// Returns a new table with rows in reverse order.
    pub fn reverse(&self) -> Result<Table> {
        let raw =
            cudf_cxx::copying::ffi::reverse_table(&self.inner).map_err(CudfError::from_cxx)?;
        Ok(Table { inner: raw })
    }

    /// Randomly sample `n` rows from this table.
    ///
    /// # Arguments
    ///
    /// * `n` - Number of rows to sample.
    /// * `with_replacement` - If true, the same row can appear multiple times.
    /// * `seed` - Random seed for reproducibility.
    ///
    /// # Errors
    ///
    /// Returns an error if `n > num_rows()` and `with_replacement` is false.
    pub fn sample(&self, n: usize, with_replacement: bool, seed: i64) -> Result<Table> {
        let raw =
            cudf_cxx::copying::ffi::sample(&self.inner, checked_i32(n)?, with_replacement, seed)
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
        let idx: Vec<i32> = indices
            .iter()
            .map(|&i| checked_i32(i))
            .collect::<Result<Vec<i32>>>()?;
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
    /// Reverse the elements of this column.
    ///
    /// Returns a new column with elements in reverse order.
    pub fn reverse(&self) -> Result<Column> {
        let raw =
            cudf_cxx::copying::ffi::reverse_column(&self.inner).map_err(CudfError::from_cxx)?;
        Ok(Column { inner: raw })
    }

    /// Shift column elements by `offset`, filling gaps with `fill_value`.
    ///
    /// A positive offset shifts elements forward (towards higher indices),
    /// a negative offset shifts backward. Elements that shift beyond the
    /// column boundaries are replaced with `fill_value`.
    ///
    /// # Errors
    ///
    /// Returns an error if `fill_value` type does not match the column type.
    pub fn shift(&self, offset: i32, fill_value: &Scalar) -> Result<Column> {
        let raw = cudf_cxx::copying::ffi::shift_column(&self.inner, offset, &fill_value.inner)
            .map_err(CudfError::from_cxx)?;
        Ok(Column { inner: raw })
    }

    /// Get a single element from this column as a [`Scalar`].
    ///
    /// # Errors
    ///
    /// Returns an error if `index` is out of bounds or a GPU error occurs.
    pub fn get_element(&self, index: usize) -> Result<Scalar> {
        let raw = cudf_cxx::copying::ffi::get_element(&self.inner, checked_i32(index)?)
            .map_err(CudfError::from_cxx)?;
        Ok(Scalar { inner: raw })
    }

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

    /// Create a column with the same type and size, with specified null mask allocation.
    ///
    /// `mask_policy`: 0=NEVER, 1=ALWAYS, 2=RETAIN.
    pub fn allocate_like(&self, mask_policy: i32) -> Result<Column> {
        let raw = cudf_cxx::copying::ffi::allocate_like(&self.inner, mask_policy)
            .map_err(CudfError::from_cxx)?;
        Ok(Column { inner: raw })
    }

    /// Select elements from a column (true in mask) or a scalar (false in mask).
    ///
    /// For each element: if `mask[i]` is true, take from `self`;
    /// if false, use `scalar`.
    pub fn copy_if_else_scalar(&self, scalar: &Scalar, mask: &Column) -> Result<Column> {
        let raw = cudf_cxx::copying::ffi::copy_if_else_col_scalar(
            &self.inner,
            &scalar.inner,
            &mask.inner,
        )
        .map_err(CudfError::from_cxx)?;
        Ok(Column { inner: raw })
    }

    /// Slice a column by pairs of `[begin, end)` indices.
    ///
    /// The `indices` slice must contain an even number of values,
    /// forming pairs: `[begin0, end0, begin1, end1, ...]`.
    /// Returns one column for each pair.
    pub fn slice_indices(&self, indices: &[usize]) -> Result<Vec<Column>> {
        let idx: Vec<i32> = indices
            .iter()
            .map(|&i| checked_i32(i))
            .collect::<Result<Vec<i32>>>()?;
        let mut result =
            cudf_cxx::copying::ffi::slice_column(&self.inner, &idx).map_err(CudfError::from_cxx)?;
        let count = cudf_cxx::copying::ffi::column_slice_result_count(&result);
        let mut columns = Vec::with_capacity(count as usize);
        for i in 0..count {
            let raw = cudf_cxx::copying::ffi::column_slice_result_get(result.pin_mut(), i)
                .map_err(CudfError::from_cxx)?;
            columns.push(Column { inner: raw });
        }
        Ok(columns)
    }

    /// Copy a range from `source` into this column (in-place).
    ///
    /// Copies elements `[source_begin, source_end)` from `source` into
    /// `self` starting at `target_begin`.
    pub fn copy_range_from(
        &mut self,
        source: &Column,
        source_begin: usize,
        source_end: usize,
        target_begin: usize,
    ) -> Result<()> {
        cudf_cxx::copying::ffi::copy_range(
            &source.inner,
            self.inner.pin_mut(),
            checked_i32(source_begin)?,
            checked_i32(source_end)?,
            checked_i32(target_begin)?,
        )
        .map_err(CudfError::from_cxx)
    }
}
