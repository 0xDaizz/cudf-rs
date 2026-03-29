//! GPU-accelerated reshape operations.
//!
//! Provides interleaving and tiling of [`Table`] columns.
//!
//! # Examples
//!
//! ```rust,no_run
//! use cudf::{Column, Table};
//!
//! let col_a = Column::from_slice(&[1i32, 2]).unwrap();
//! let col_b = Column::from_slice(&[3i32, 4]).unwrap();
//! let table = Table::new(vec![col_a, col_b]).unwrap();
//!
//! let interleaved = table.interleave_columns().unwrap();
//! // Result: [1, 3, 2, 4]
//! ```

use crate::column::Column;
use crate::error::{CudfError, Result};
use crate::table::Table;

impl Table {
    /// Interleave all columns into a single column.
    ///
    /// Columns must all have the same data type. Elements are taken
    /// round-robin from each column: `[col0[0], col1[0], col0[1], col1[1], ...]`.
    pub fn interleave_columns(&self) -> Result<Column> {
        let raw = cudf_cxx::reshape::ffi::interleave_columns(&self.inner)
            .map_err(CudfError::from_cxx)?;
        Ok(Column { inner: raw })
    }

    /// Tile (repeat) this table's rows the specified number of times.
    ///
    /// Returns a new table with `num_rows * count` rows.
    pub fn tile(&self, count: usize) -> Result<Table> {
        let raw = cudf_cxx::reshape::ffi::tile(&self.inner, count as i32)
            .map_err(CudfError::from_cxx)?;
        Ok(Table { inner: raw })
    }
}
