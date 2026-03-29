//! SQL LIKE pattern matching operations.

use crate::column::Column;
use crate::error::{CudfError, Result};

impl Column {
    /// SQL LIKE pattern matching.
    ///
    /// `%` matches zero or more characters, `_` matches any single character.
    /// An optional `escape_char` can be specified to escape wildcards.
    pub fn str_like(&self, pattern: &str, escape_char: &str) -> Result<Column> {
        let result = cudf_cxx::strings::like::ffi::str_like(&self.inner, pattern, escape_char)
            .map_err(CudfError::from_cxx)?;
        Ok(Column { inner: result })
    }
}
