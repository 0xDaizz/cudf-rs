//! Regex-based string split operations.

use crate::column::Column;
use crate::error::{CudfError, Result};
use crate::table::Table;

impl Column {
    /// Split each string by regex `pattern`, returning a table of string columns.
    ///
    /// `maxsplit` limits the number of splits (-1 for unlimited).
    pub fn str_split_re(&self, pattern: &str, maxsplit: i32) -> Result<Table> {
        let result = cudf_cxx::strings::split_re::ffi::str_split_re(&self.inner, pattern, maxsplit)
            .map_err(CudfError::from_cxx)?;
        Ok(Table { inner: result })
    }

    /// Split each string by regex `pattern` from the right, returning a table.
    ///
    /// `maxsplit` limits the number of splits (-1 for unlimited).
    pub fn str_rsplit_re(&self, pattern: &str, maxsplit: i32) -> Result<Table> {
        let result =
            cudf_cxx::strings::split_re::ffi::str_rsplit_re(&self.inner, pattern, maxsplit)
                .map_err(CudfError::from_cxx)?;
        Ok(Table { inner: result })
    }

    /// Split each string by regex, returning a list column of strings.
    ///
    /// `maxsplit` limits the number of splits (-1 for unlimited).
    pub fn str_split_record_re(&self, pattern: &str, maxsplit: i32) -> Result<Column> {
        let result =
            cudf_cxx::strings::split_re::ffi::str_split_record_re(&self.inner, pattern, maxsplit)
                .map_err(CudfError::from_cxx)?;
        Ok(Column { inner: result })
    }

    /// Split each string from the right by regex, returning a list column.
    ///
    /// `maxsplit` limits the number of splits (-1 for unlimited).
    pub fn str_rsplit_record_re(&self, pattern: &str, maxsplit: i32) -> Result<Column> {
        let result =
            cudf_cxx::strings::split_re::ffi::str_rsplit_record_re(&self.inner, pattern, maxsplit)
                .map_err(CudfError::from_cxx)?;
        Ok(Column { inner: result })
    }
}
