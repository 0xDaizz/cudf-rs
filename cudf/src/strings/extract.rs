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

    /// Extract all matches of capture groups per row, returning a list column.
    ///
    /// Each row in the output lists column contains all captured substrings
    /// from all matches found in the corresponding input string.
    pub fn str_extract_all_record(&self, pattern: &str) -> Result<Column> {
        let result = cudf_cxx::strings::extract::ffi::str_extract_all_record(&self.inner, pattern)
            .map_err(CudfError::from_cxx)?;
        Ok(Column { inner: result })
    }

    /// Extract a single capture group from each string matching `pattern`.
    ///
    /// `group_index` specifies which capture group to return (0-based).
    /// Returns a string column with the extracted group value, or null
    /// if the pattern doesn't match.
    pub fn str_extract_single(&self, pattern: &str, group_index: i32) -> Result<Column> {
        let result =
            cudf_cxx::strings::extract::ffi::str_extract_single(&self.inner, pattern, group_index)
                .map_err(CudfError::from_cxx)?;
        Ok(Column { inner: result })
    }
}
