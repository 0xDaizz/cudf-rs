//! GPU-accelerated list column operations.
//!
//! Provides operations on list (nested) columns: exploding lists into rows,
//! sorting elements within lists, checking containment, and extracting elements.
//!
//! # Examples
//!
//! ```rust,no_run
//! use cudf::Column;
//!
//! // Extract the first element from each list row
//! let list_col: Column = todo!("create a list column");
//! let first_elements = list_col.lists_extract(0).unwrap();
//! ```

use crate::column::Column;
use crate::error::{CudfError, Result};
use crate::scalar::Scalar;
use crate::sorting::NullOrder;
use crate::table::Table;

impl Table {
    /// Explode a list column, expanding each list element into its own row.
    ///
    /// The column at `explode_col_idx` must be a list column. Each element
    /// of the list becomes a separate row in the output table. Null lists
    /// are dropped.
    ///
    /// # Errors
    ///
    /// Returns an error if the column index is out of bounds, the column is
    /// not a list type, or a GPU error occurs.
    pub fn lists_explode(&self, explode_col_idx: usize) -> Result<Table> {
        let raw = cudf_cxx::lists::ops::ffi::lists_explode(&self.inner, explode_col_idx as i32)
            .map_err(CudfError::from_cxx)?;
        Ok(Table { inner: raw })
    }

    /// Explode a list column, retaining null entries and empty lists as null rows.
    ///
    /// Like [`lists_explode`](Self::lists_explode) but null and empty lists
    /// produce a null row instead of being dropped.
    pub fn lists_explode_outer(&self, explode_col_idx: usize) -> Result<Table> {
        let raw =
            cudf_cxx::lists::ops::ffi::lists_explode_outer(&self.inner, explode_col_idx as i32)
                .map_err(CudfError::from_cxx)?;
        Ok(Table { inner: raw })
    }
}

impl Column {
    /// Sort elements within each list row.
    ///
    /// Returns a new list column where the elements within each row are
    /// sorted according to the given order and null placement.
    pub fn lists_sort(&self, ascending: bool, null_order: NullOrder) -> Result<Column> {
        let raw = cudf_cxx::lists::ops::ffi::lists_sort(&self.inner, ascending, null_order as i32)
            .map_err(CudfError::from_cxx)?;
        Ok(Column { inner: raw })
    }

    /// Check whether each list row contains the given scalar value.
    ///
    /// Returns a boolean column where `true` indicates the list row
    /// contains the search key.
    pub fn lists_contains(&self, search_key: &Scalar) -> Result<Column> {
        let raw = cudf_cxx::lists::ops::ffi::lists_contains(&self.inner, &search_key.inner)
            .map_err(CudfError::from_cxx)?;
        Ok(Column { inner: raw })
    }

    /// Check whether each list row contains any null elements.
    ///
    /// Returns a boolean column where `true` indicates the list row
    /// has at least one null element.
    pub fn lists_contains_nulls(&self) -> Result<Column> {
        let raw = cudf_cxx::lists::ops::ffi::lists_contains_nulls(&self.inner)
            .map_err(CudfError::from_cxx)?;
        Ok(Column { inner: raw })
    }

    /// Extract the element at `index` from each list row.
    ///
    /// Negative indices count from the end of each list. If `index` is
    /// out of bounds for a given row, the output for that row is null.
    pub fn lists_extract(&self, index: i32) -> Result<Column> {
        let raw = cudf_cxx::lists::ops::ffi::lists_extract(&self.inner, index)
            .map_err(CudfError::from_cxx)?;
        Ok(Column { inner: raw })
    }
}
