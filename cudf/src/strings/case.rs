//! String case conversion operations.
//!
//! # Examples
//!
//! ```rust,no_run
//! use cudf::Column;
//!
//! // Assuming `col` is a STRING column
//! let upper = col.str_to_upper().unwrap();
//! let lower = col.str_to_lower().unwrap();
//! ```

use crate::column::Column;
use crate::error::{CudfError, Result};

impl Column {
    /// Convert all characters in each string to uppercase.
    pub fn str_to_upper(&self) -> Result<Column> {
        let result =
            cudf_cxx::strings::case::ffi::str_to_upper(&self.inner).map_err(CudfError::from_cxx)?;
        Ok(Column { inner: result })
    }

    /// Convert all characters in each string to lowercase.
    pub fn str_to_lower(&self) -> Result<Column> {
        let result =
            cudf_cxx::strings::case::ffi::str_to_lower(&self.inner).map_err(CudfError::from_cxx)?;
        Ok(Column { inner: result })
    }

    /// Swap the case of all characters in each string.
    pub fn str_swapcase(&self) -> Result<Column> {
        let result =
            cudf_cxx::strings::case::ffi::str_swapcase(&self.inner).map_err(CudfError::from_cxx)?;
        Ok(Column { inner: result })
    }
}
