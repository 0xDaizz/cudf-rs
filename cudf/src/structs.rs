//! GPU-accelerated struct column operations.
//!
//! Provides operations on struct (nested) columns, such as extracting
//! child columns by index.
//!
//! # Examples
//!
//! ```rust,no_run
//! use cudf::Column;
//!
//! // Struct columns are created via I/O readers (Parquet, JSON)
//! // or the Arrow interop interface.
//! // let struct_col: Column = /* from parquet/json/arrow */;
//! // let child = struct_col.structs_extract(0).unwrap();
//! ```

use crate::column::Column;
use crate::error::{CudfError, Result};

impl Column {
    /// Extract the child column at `index` from a struct column.
    ///
    /// Returns a copy of the child column. The index is zero-based.
    ///
    /// # Errors
    ///
    /// Returns an error if this column is not a struct type, the index
    /// is out of bounds, or a GPU error occurs.
    pub fn structs_extract(&self, index: i32) -> Result<Column> {
        let raw = cudf_cxx::structs::ffi::structs_extract(&self.inner, index)
            .map_err(CudfError::from_cxx)?;
        Ok(Column { inner: raw })
    }
}
