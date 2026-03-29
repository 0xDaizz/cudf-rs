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

    /// Capitalize the first character of each string (or each word).
    ///
    /// If `delimiters` is empty, only the first character of each string
    /// is capitalized. Otherwise, a non-delimiter character is capitalized
    /// after any delimiter character.
    pub fn str_capitalize(&self, delimiters: &str) -> Result<Column> {
        let result = cudf_cxx::strings::case::ffi::str_capitalize(&self.inner, delimiters)
            .map_err(CudfError::from_cxx)?;
        Ok(Column { inner: result })
    }

    /// Convert each string to title case.
    ///
    /// The first character of each word is upper-cased, and the rest
    /// are lower-cased. A word is a sequence of alpha characters.
    pub fn str_title(&self) -> Result<Column> {
        let result =
            cudf_cxx::strings::case::ffi::str_title(&self.inner).map_err(CudfError::from_cxx)?;
        Ok(Column { inner: result })
    }

    /// Check if each string is in title case.
    ///
    /// Returns a BOOL8 column where `true` indicates the string
    /// is in title format (first char of each word is upper-case,
    /// rest are lower-case).
    pub fn str_is_title(&self) -> Result<Column> {
        let result =
            cudf_cxx::strings::case::ffi::str_is_title(&self.inner).map_err(CudfError::from_cxx)?;
        Ok(Column { inner: result })
    }
}
