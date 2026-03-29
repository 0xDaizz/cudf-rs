//! String findall (regex match extraction) operations.

use crate::column::Column;
use crate::error::{CudfError, Result};

impl Column {
    /// Find all occurrences of `pattern` in each string.
    ///
    /// Returns a lists column where each row contains all matches.
    pub fn str_findall(&self, pattern: &str) -> Result<Column> {
        let result = cudf_cxx::strings::findall::ffi::str_findall(&self.inner, pattern)
            .map_err(CudfError::from_cxx)?;
        Ok(Column { inner: result })
    }

    /// Find starting position of the first regex match in each string.
    ///
    /// Returns an INT32 column (-1 if not found).
    pub fn str_find_re(&self, pattern: &str) -> Result<Column> {
        let result = cudf_cxx::strings::findall::ffi::str_find_re(&self.inner, pattern)
            .map_err(CudfError::from_cxx)?;
        Ok(Column { inner: result })
    }
}
