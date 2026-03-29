//! GPU-accelerated table transposition.
//!
//! Swaps rows and columns of a [`Table`].
//!
//! # Examples
//!
//! ```rust,no_run
//! use cudf::{Column, Table};
//!
//! let col_a = Column::from_slice(&[1i32, 2, 3]).unwrap();
//! let col_b = Column::from_slice(&[4i32, 5, 6]).unwrap();
//! let table = Table::new(vec![col_a, col_b]).unwrap();
//!
//! let transposed = table.transpose().unwrap();
//! assert_eq!(transposed.num_columns(), 3);
//! assert_eq!(transposed.num_rows(), 2);
//! ```

use crate::error::{CudfError, Result};
use crate::table::Table;

impl Table {
    /// Transpose this table (swap rows and columns).
    ///
    /// All columns must have the same data type. The resulting table
    /// will have `num_rows` columns and `num_columns` rows.
    pub fn transpose(&self) -> Result<Table> {
        let raw =
            cudf_cxx::transpose::ffi::transpose_table(&self.inner).map_err(CudfError::from_cxx)?;
        Ok(Table { inner: raw })
    }
}
