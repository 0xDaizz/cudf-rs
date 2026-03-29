//! String character type checking operations.

use crate::column::Column;
use crate::error::{CudfError, Result};

/// Character type bitmask values for `str_all_characters_of_type`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StringCharacterType {
    Decimal = 1,
    Numeric = 2,
    Digit = 4,
    Alpha = 8,
    Space = 16,
    Upper = 32,
    Lower = 64,
    Alphanum = 10, // ALPHA | NUMERIC
    AllTypes = 127,
}

impl Column {
    /// Check if all characters of each string are of the given `types`.
    ///
    /// `verify_types` specifies which character types to include in the check.
    /// Returns a BOOL8 column.
    pub fn str_all_characters_of_type(&self, types: u32, verify_types: u32) -> Result<Column> {
        let result = cudf_cxx::strings::char_types::ffi::str_all_characters_of_type(
            &self.inner,
            types,
            verify_types,
        )
        .map_err(CudfError::from_cxx)?;
        Ok(Column { inner: result })
    }
}
