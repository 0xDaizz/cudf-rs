//! String repeat operations.

use crate::column::Column;
use crate::error::{CudfError, Result};

impl Column {
    /// Repeat each string `count` times.
    pub fn str_repeat(&self, count: i32) -> Result<Column> {
        let result = cudf_cxx::strings::repeat::ffi::str_repeat(&self.inner, count)
            .map_err(CudfError::from_cxx)?;
        Ok(Column { inner: result })
    }
}
