//! String reverse operations.

use crate::column::Column;
use crate::error::{CudfError, Result};

impl Column {
    /// Reverse each string character-by-character.
    pub fn str_reverse(&self) -> Result<Column> {
        let result = cudf_cxx::strings::reverse::ffi::str_reverse(&self.inner)
            .map_err(CudfError::from_cxx)?;
        Ok(Column { inner: result })
    }
}
