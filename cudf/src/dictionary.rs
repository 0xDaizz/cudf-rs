//! GPU-accelerated dictionary encoding and decoding.
//!
//! Dictionary encoding replaces repeated values with integer indices into
//! a sorted key table, which can significantly reduce memory usage and
//! improve performance for columns with low cardinality.
//!
//! # Examples
//!
//! ```rust,no_run
//! use cudf::Column;
//!
//! let col = Column::from_slice(&[1i32, 2, 1, 2, 1]).unwrap();
//! let encoded = col.dictionary_encode().unwrap();
//! let decoded = encoded.dictionary_decode().unwrap();
//! ```

use crate::column::Column;
use crate::error::{CudfError, Result};

impl Column {
    /// Dictionary-encode this column.
    ///
    /// Returns a DICTIONARY type column with sorted unique keys and
    /// integer indices. Null values are preserved in the output.
    ///
    /// # Errors
    ///
    /// Returns an error if the column is already dictionary-encoded
    /// or a GPU error occurs.
    pub fn dictionary_encode(&self) -> Result<Column> {
        let raw = cudf_cxx::dictionary::ffi::dictionary_encode(&self.inner)
            .map_err(CudfError::from_cxx)?;
        Ok(Column { inner: raw })
    }

    /// Decode a dictionary-encoded column back to its original type.
    ///
    /// Gathers keys using the indices to reconstruct the original values.
    ///
    /// # Errors
    ///
    /// Returns an error if this column is not dictionary-encoded
    /// or a GPU error occurs.
    pub fn dictionary_decode(&self) -> Result<Column> {
        let raw = cudf_cxx::dictionary::ffi::dictionary_decode(&self.inner)
            .map_err(CudfError::from_cxx)?;
        Ok(Column { inner: raw })
    }
}
