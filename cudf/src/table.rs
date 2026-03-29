//! GPU-resident table type.
//!
//! A [`Table`] is an ordered collection of [`Column`]s, analogous to a DataFrame.
//! It owns GPU memory for all its columns and frees it on drop.
//!
//! # Examples
//!
//! ```rust,no_run
//! use cudf::{Column, Table};
//!
//! let col_a = Column::from_slice(&[1i32, 2, 3]).unwrap();
//! let col_b = Column::from_slice(&[4.0f64, 5.0, 6.0]).unwrap();
//! let table = Table::new(vec![col_a, col_b]).unwrap();
//!
//! assert_eq!(table.num_columns(), 2);
//! assert_eq!(table.num_rows(), 3);
//! ```

use std::fmt;

use cxx::UniquePtr;

use crate::column::Column;
use crate::error::{CudfError, Result};
use crate::types::checked_i32;

/// An owning, GPU-resident table (ordered collection of columns).
///
/// `Table` wraps a `std::unique_ptr<cudf::table>` on the C++ side.
/// Dropping a `Table` frees all associated GPU memory.
///
/// # Thread Safety
///
/// Like [`Column`], `Table` implements [`Send`] but not [`Sync`].
pub struct Table {
    pub(crate) inner: UniquePtr<cudf_cxx::table::ffi::OwnedTable>,
}

// SAFETY: Same reasoning as Column -- GPU memory is process-global.
unsafe impl Send for Table {}

impl Table {
    // -- Accessors --

    /// Number of columns in this table.
    pub fn num_columns(&self) -> usize {
        self.inner.num_columns() as usize
    }

    /// Number of rows in this table.
    ///
    /// Returns 0 if the table's columns have been released (e.g., after
    /// `table_release_column` has consumed all columns on the C++ side).
    pub fn num_rows(&self) -> usize {
        self.inner.num_rows() as usize
    }

    /// Whether this table has zero rows.
    pub fn is_empty(&self) -> bool {
        self.num_rows() == 0
    }

    // -- Construction --

    /// Create a table from a vector of columns.
    ///
    /// The columns are consumed (moved to GPU). All columns must have
    /// the same number of rows.
    ///
    /// # Errors
    ///
    /// Returns an error if column lengths don't match or if a GPU error occurs.
    pub fn new(columns: Vec<Column>) -> Result<Self> {
        // Allow empty tables (libcudf supports them)
        if columns.is_empty() {
            let mut builder = cudf_cxx::table::ffi::table_builder_new();
            let raw = builder.pin_mut().build().map_err(CudfError::from_cxx)?;
            return Ok(Self { inner: raw });
        }

        // Check all columns have the same length
        let expected_len = columns[0].len();
        for (i, col) in columns.iter().enumerate().skip(1) {
            if col.len() != expected_len {
                return Err(CudfError::InvalidArgument(format!(
                    "Column {} has {} rows, expected {} (matching column 0)",
                    i,
                    col.len(),
                    expected_len
                )));
            }
        }

        let mut builder = cudf_cxx::table::ffi::table_builder_new();
        for col in columns {
            builder.pin_mut().add_column(col.inner);
        }
        let raw = builder.pin_mut().build().map_err(CudfError::from_cxx)?;

        Ok(Self { inner: raw })
    }

    // -- Column Access --

    /// Get a copy of the column at the given index.
    ///
    /// This creates a deep copy -- the table retains its own copy of the data.
    /// For zero-copy access, use column views (available in later phases).
    ///
    /// # Errors
    ///
    /// Returns `CudfError::IndexOutOfBounds` if `index >= num_columns()`.
    pub fn column(&self, index: usize) -> Result<Column> {
        if index >= self.num_columns() {
            return Err(CudfError::IndexOutOfBounds {
                index,
                size: self.num_columns(),
            });
        }

        let index_i32 = checked_i32(index)?;
        let raw = cudf_cxx::table::ffi::table_get_column(&self.inner, index_i32)
            .map_err(CudfError::from_cxx)?;

        Ok(Column { inner: raw })
    }

    /// Decompose the table into its constituent columns, consuming it.
    ///
    /// This is a zero-copy operation -- the columns take ownership of
    /// the GPU memory that the table previously owned.
    pub fn into_columns(mut self) -> Result<Vec<Column>> {
        let n = self.num_columns();
        let mut result = Vec::with_capacity(n);
        // Release in reverse order to avoid index shifting
        for i in (0..n).rev() {
            let col = cudf_cxx::table::ffi::table_release_column(self.inner.pin_mut(), i as i32)
                .map_err(CudfError::from_cxx)?;
            result.push(Column { inner: col });
        }
        result.reverse();
        Ok(result)
    }
}

/// A table together with column names, as returned by IO readers.
///
/// This wraps a [`Table`] with the column names extracted from the
/// file metadata (e.g., Parquet schema, CSV header, JSON keys).
pub struct TableWithMetadata {
    /// The GPU-resident table data.
    pub table: Table,
    /// Column names corresponding to each column in `table`.
    pub column_names: Vec<String>,
}

impl TableWithMetadata {
    /// Build from the raw FFI metadata wrapper.
    pub(crate) fn from_raw(
        raw: cxx::UniquePtr<cudf_cxx::table::ffi::OwnedTableWithMetadata>,
    ) -> Result<Self> {
        let ncols = cudf_cxx::table::ffi::table_meta_num_columns(&raw);
        let mut names = Vec::with_capacity(ncols as usize);
        for i in 0..ncols {
            let name = cudf_cxx::table::ffi::table_meta_column_name(&raw, i)
                .map_err(CudfError::from_cxx)?;
            names.push(name);
        }
        let inner =
            cudf_cxx::table::ffi::table_meta_into_table(raw).map_err(CudfError::from_cxx)?;
        Ok(Self {
            table: Table { inner },
            column_names: names,
        })
    }
}

impl fmt::Debug for TableWithMetadata {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "TableWithMetadata(columns={}, rows={}, names={:?})",
            self.table.num_columns(),
            self.table.num_rows(),
            self.column_names
        )
    }
}

impl fmt::Display for TableWithMetadata {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "TableWithMetadata(columns={}, rows={}, names={:?})",
            self.table.num_columns(),
            self.table.num_rows(),
            self.column_names
        )
    }
}

impl fmt::Debug for Table {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl fmt::Display for Table {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Table(columns={}, rows={})",
            self.num_columns(),
            self.num_rows()
        )
    }
}
