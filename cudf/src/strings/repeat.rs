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

    /// Repeat each string by a per-row count from `counts` column.
    ///
    /// `counts` must be an integer column with the same number of rows.
    /// A non-positive count produces an empty string. Null rows in either
    /// column produce null output.
    pub fn str_repeat_per_row(&self, counts: &Column) -> Result<Column> {
        let result = cudf_cxx::strings::repeat::ffi::str_repeat_per_row(&self.inner, &counts.inner)
            .map_err(CudfError::from_cxx)?;
        Ok(Column { inner: result })
    }
}
