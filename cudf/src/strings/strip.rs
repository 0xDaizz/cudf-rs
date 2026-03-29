//! String strip (trim) operations.

use crate::column::Column;
use crate::error::{CudfError, Result};

impl Column {
    /// Strip leading and trailing whitespace from each string.
    pub fn str_strip(&self) -> Result<Column> {
        let result = cudf_cxx::strings::strip::ffi::str_strip(&self.inner)
            .map_err(CudfError::from_cxx)?;
        Ok(Column { inner: result })
    }

    /// Strip leading whitespace from each string.
    pub fn str_lstrip(&self) -> Result<Column> {
        let result = cudf_cxx::strings::strip::ffi::str_lstrip(&self.inner)
            .map_err(CudfError::from_cxx)?;
        Ok(Column { inner: result })
    }

    /// Strip trailing whitespace from each string.
    pub fn str_rstrip(&self) -> Result<Column> {
        let result = cudf_cxx::strings::strip::ffi::str_rstrip(&self.inner)
            .map_err(CudfError::from_cxx)?;
        Ok(Column { inner: result })
    }
}
