//! Filling and sequence generation operations.
//!
//! Provides GPU-accelerated fill, repeat, and sequence generation
//! for columns and tables.
//!
//! # Examples
//!
//! ```rust,no_run
//! use cudf::filling;
//!
//! // Generate a sequence 0, 1, 2, ..., 9
//! let seq = filling::sequence_i32(10, 0, 1).unwrap();
//! assert_eq!(seq.len(), 10);
//! ```

use crate::column::Column;
use crate::error::{CudfError, Result};
use crate::table::Table;
use crate::types::checked_i32;

/// Generate an i32 sequence: init, init+step, init+2*step, ...
///
/// # Examples
///
/// ```rust,no_run
/// use cudf::filling;
///
/// let col = filling::sequence_i32(5, 10, 2).unwrap();
/// // col contains: [10, 12, 14, 16, 18]
/// ```
pub fn sequence_i32(size: usize, init: i32, step: i32) -> Result<Column> {
    let raw = cudf_cxx::filling::ffi::sequence_i32(checked_i32(size)?, init, step)
        .map_err(CudfError::from_cxx)?;
    Ok(Column { inner: raw })
}

/// Generate an i64 sequence: init, init+step, init+2*step, ...
pub fn sequence_i64(size: usize, init: i64, step: i64) -> Result<Column> {
    let raw = cudf_cxx::filling::ffi::sequence_i64(checked_i32(size)?, init, step)
        .map_err(CudfError::from_cxx)?;
    Ok(Column { inner: raw })
}

/// Generate an f32 sequence: init, init+step, init+2*step, ...
pub fn sequence_f32(size: usize, init: f32, step: f32) -> Result<Column> {
    let raw = cudf_cxx::filling::ffi::sequence_f32(checked_i32(size)?, init, step)
        .map_err(CudfError::from_cxx)?;
    Ok(Column { inner: raw })
}

/// Generate an f64 sequence: init, init+step, init+2*step, ...
pub fn sequence_f64(size: usize, init: f64, step: f64) -> Result<Column> {
    let raw = cudf_cxx::filling::ffi::sequence_f64(checked_i32(size)?, init, step)
        .map_err(CudfError::from_cxx)?;
    Ok(Column { inner: raw })
}

impl Table {
    /// Repeat all rows of this table `count` times.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use cudf::{Column, Table};
    ///
    /// let col = Column::from_slice(&[1i32, 2, 3]).unwrap();
    /// let table = Table::new(vec![col]).unwrap();
    /// let repeated = table.repeat(3).unwrap();
    /// assert_eq!(repeated.num_rows(), 9);
    /// ```
    pub fn repeat(&self, count: usize) -> Result<Table> {
        let raw = cudf_cxx::filling::ffi::repeat_table(&self.inner, checked_i32(count)?)
            .map_err(CudfError::from_cxx)?;
        Ok(Table { inner: raw })
    }

    /// Repeat rows of this table, where each row is repeated by the
    /// corresponding value in the `counts` column.
    ///
    /// The `counts` column must be an integer type with the same number
    /// of rows as the table.
    pub fn repeat_variable(&self, counts: &Column) -> Result<Table> {
        let raw = cudf_cxx::filling::ffi::repeat_table_variable(&self.inner, &counts.inner)
            .map_err(CudfError::from_cxx)?;
        Ok(Table { inner: raw })
    }
}

/// Generate a sequence of timestamps separated by a fixed number of months.
///
/// The `init` scalar must be a timestamp type. Returns a column of
/// timestamps offset from `init` by multiples of `months`.
pub fn calendrical_month_sequence(
    size: usize,
    init: &crate::scalar::Scalar,
    months: i32,
) -> Result<Column> {
    let raw = cudf_cxx::filling::ffi::calendrical_month_sequence(checked_i32(size)?, &init.inner, months)
        .map_err(CudfError::from_cxx)?;
    Ok(Column { inner: raw })
}
