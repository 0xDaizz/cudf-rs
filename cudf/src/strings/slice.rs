//! String slicing (substring) operations.

use crate::column::Column;
use crate::error::{CudfError, Result};

impl Column {
    /// Extract a substring from each string, from `start` to `stop` (exclusive).
    ///
    /// Use `stop = -1` to slice to end of string.
    pub fn str_slice(&self, start: i32, stop: i32) -> Result<Column> {
        let result = cudf_cxx::strings::slice::ffi::str_slice(&self.inner, start, stop)
            .map_err(CudfError::from_cxx)?;
        Ok(Column { inner: result })
    }
}
