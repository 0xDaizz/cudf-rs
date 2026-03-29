//! String concatenation (join) operations.

use crate::column::Column;
use crate::error::{CudfError, Result};

impl Column {
    /// Concatenate all strings in the column into a single string,
    /// separated by `separator`.
    ///
    /// Returns a single-element string column.
    pub fn str_join(&self, separator: &str) -> Result<Column> {
        let result = cudf_cxx::strings::combine::ffi::str_join(&self.inner, separator)
            .map_err(CudfError::from_cxx)?;
        Ok(Column { inner: result })
    }

    /// Join list elements within each row using `separator`.
    ///
    /// The input must be a lists column of strings. Each row's list elements
    /// are concatenated with the separator between them.
    pub fn str_join_list_elements(&self, separator: &str) -> Result<Column> {
        let result =
            cudf_cxx::strings::combine::ffi::str_join_list_elements(&self.inner, separator)
                .map_err(CudfError::from_cxx)?;
        Ok(Column { inner: result })
    }
}
