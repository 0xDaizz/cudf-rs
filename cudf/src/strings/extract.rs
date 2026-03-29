//! String regex extract operations.

use crate::column::Column;
use crate::error::{CudfError, Result};
use crate::table::Table;

impl Column {
    /// Extract capture groups from each string matching `pattern`.
    ///
    /// Returns a table with one column per capture group.
    pub fn str_extract(&self, pattern: &str) -> Result<Table> {
        let result = cudf_cxx::strings::extract::ffi::str_extract(&self.inner, pattern)
            .map_err(CudfError::from_cxx)?;
        Ok(Table { inner: result })
    }
}
