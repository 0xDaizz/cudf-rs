//! String attributes operations (character count, byte count, code points).

use crate::column::Column;
use crate::error::{CudfError, Result};

impl Column {
    /// Count the number of characters in each string.
    pub fn str_count_characters(&self) -> Result<Column> {
        let result = cudf_cxx::strings::attributes::ffi::str_count_characters(&self.inner)
            .map_err(CudfError::from_cxx)?;
        Ok(Column { inner: result })
    }

    /// Count the number of bytes in each string.
    pub fn str_count_bytes(&self) -> Result<Column> {
        let result = cudf_cxx::strings::attributes::ffi::str_count_bytes(&self.inner)
            .map_err(CudfError::from_cxx)?;
        Ok(Column { inner: result })
    }

    /// Return the code points for each character of each string.
    pub fn str_code_points(&self) -> Result<Column> {
        let result = cudf_cxx::strings::attributes::ffi::str_code_points(&self.inner)
            .map_err(CudfError::from_cxx)?;
        Ok(Column { inner: result })
    }
}
