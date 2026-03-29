//! String wrap operations.

use crate::column::Column;
use crate::error::{CudfError, Result};

impl Column {
    /// Wrap long strings by inserting newlines at whitespace boundaries.
    ///
    /// Lines will be no longer than `width` characters where possible.
    pub fn str_wrap(&self, width: i32) -> Result<Column> {
        let result = cudf_cxx::strings::wrap::ffi::str_wrap(&self.inner, width)
            .map_err(CudfError::from_cxx)?;
        Ok(Column { inner: result })
    }
}
