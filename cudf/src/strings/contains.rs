//! String contains/match operations.

use crate::column::Column;
use crate::error::{CudfError, Result};

impl Column {
    /// Check if each string contains the literal `target`.
    ///
    /// Returns a BOOL8 column.
    pub fn str_contains(&self, target: &str) -> Result<Column> {
        let result =
            cudf_cxx::strings::contains::ffi::str_contains(&self.inner, target)
                .map_err(CudfError::from_cxx)?;
        Ok(Column { inner: result })
    }

    /// Check if each string contains a match for the regex `pattern`.
    ///
    /// Returns a BOOL8 column.
    pub fn str_contains_re(&self, pattern: &str) -> Result<Column> {
        let result =
            cudf_cxx::strings::contains::ffi::str_contains_re(&self.inner, pattern)
                .map_err(CudfError::from_cxx)?;
        Ok(Column { inner: result })
    }

    /// Check if each string fully matches the regex `pattern`.
    ///
    /// Returns a BOOL8 column.
    pub fn str_matches_re(&self, pattern: &str) -> Result<Column> {
        let result =
            cudf_cxx::strings::contains::ffi::str_matches_re(&self.inner, pattern)
                .map_err(CudfError::from_cxx)?;
        Ok(Column { inner: result })
    }

    /// Count non-overlapping occurrences of the regex `pattern` in each string.
    ///
    /// Returns an INT32 column.
    pub fn str_count_re(&self, pattern: &str) -> Result<Column> {
        let result =
            cudf_cxx::strings::contains::ffi::str_count_re(&self.inner, pattern)
                .map_err(CudfError::from_cxx)?;
        Ok(Column { inner: result })
    }
}
