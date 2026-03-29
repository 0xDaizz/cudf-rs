//! String replace operations.

use crate::column::Column;
use crate::error::{CudfError, Result};

impl Column {
    /// Replace all occurrences of `target` with `replacement` in each string.
    pub fn str_replace(&self, target: &str, replacement: &str) -> Result<Column> {
        let result = cudf_cxx::strings::replace::ffi::str_replace(&self.inner, target, replacement)
            .map_err(CudfError::from_cxx)?;
        Ok(Column { inner: result })
    }

    /// Replace all regex matches of `pattern` with `replacement` in each string.
    pub fn str_replace_re(&self, pattern: &str, replacement: &str) -> Result<Column> {
        let result =
            cudf_cxx::strings::replace::ffi::str_replace_re(&self.inner, pattern, replacement)
                .map_err(CudfError::from_cxx)?;
        Ok(Column { inner: result })
    }
}
