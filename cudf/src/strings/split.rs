//! String split operations.

use crate::column::Column;
use crate::error::{CudfError, Result};
use crate::table::Table;

impl Column {
    /// Split each string by `delimiter`, returning a table of string columns.
    ///
    /// `maxsplit` limits the number of splits (-1 for unlimited).
    pub fn str_split(&self, delimiter: &str, maxsplit: i32) -> Result<Table> {
        let result = cudf_cxx::strings::split::ffi::str_split(&self.inner, delimiter, maxsplit)
            .map_err(CudfError::from_cxx)?;
        Ok(Table { inner: result })
    }

    /// Split each string by `delimiter` from the right, returning a table.
    ///
    /// `maxsplit` limits the number of splits (-1 for unlimited).
    pub fn str_rsplit(&self, delimiter: &str, maxsplit: i32) -> Result<Table> {
        let result = cudf_cxx::strings::split::ffi::str_rsplit(&self.inner, delimiter, maxsplit)
            .map_err(CudfError::from_cxx)?;
        Ok(Table { inner: result })
    }

    /// Split each string by `delimiter`, returning a list column of strings.
    ///
    /// `maxsplit` limits the number of splits (-1 for unlimited).
    pub fn str_split_record(&self, delimiter: &str, maxsplit: i32) -> Result<Column> {
        let result =
            cudf_cxx::strings::split::ffi::str_split_record(&self.inner, delimiter, maxsplit)
                .map_err(CudfError::from_cxx)?;
        Ok(Column { inner: result })
    }

    /// Split each string from the right by `delimiter`, returning a list column.
    ///
    /// `maxsplit` limits the number of splits (-1 for unlimited).
    pub fn str_rsplit_record(&self, delimiter: &str, maxsplit: i32) -> Result<Column> {
        let result =
            cudf_cxx::strings::split::ffi::str_rsplit_record(&self.inner, delimiter, maxsplit)
                .map_err(CudfError::from_cxx)?;
        Ok(Column { inner: result })
    }

    /// Return a single part from splitting each string by `delimiter`.
    ///
    /// `index` is the 0-based part index to return. If the string has fewer
    /// parts, the output for that row is null.
    pub fn str_split_part(&self, delimiter: &str, index: i32) -> Result<Column> {
        let result = cudf_cxx::strings::split::ffi::str_split_part(&self.inner, delimiter, index)
            .map_err(CudfError::from_cxx)?;
        Ok(Column { inner: result })
    }
}
