//! String padding operations.

use crate::column::Column;
use crate::error::{CudfError, Result};

/// Side for padding operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PadSide {
    /// Pad on the left (right-justify).
    Left = 0,
    /// Pad on the right (left-justify).
    Right = 1,
    /// Pad on both sides (center).
    Both = 2,
}

impl Column {
    /// Pad each string to at least `width` characters using `fill_char`.
    pub fn str_pad(&self, width: i32, side: PadSide, fill_char: &str) -> Result<Column> {
        let result =
            cudf_cxx::strings::padding::ffi::str_pad(&self.inner, width, side as i32, fill_char)
                .map_err(CudfError::from_cxx)?;
        Ok(Column { inner: result })
    }

    /// Pad each string with leading zeros to at least `width` characters.
    pub fn str_zfill(&self, width: i32) -> Result<Column> {
        let result = cudf_cxx::strings::padding::ffi::str_zfill(&self.inner, width)
            .map_err(CudfError::from_cxx)?;
        Ok(Column { inner: result })
    }

    /// Zero-fill with per-row widths from an integer column.
    ///
    /// Each string is padded with leading zeros to the width specified
    /// by the corresponding element in the `widths` column.
    pub fn str_zfill_by_widths(&self, widths: &Column) -> Result<Column> {
        let result =
            cudf_cxx::strings::padding::ffi::str_zfill_by_widths(&self.inner, &widths.inner)
                .map_err(CudfError::from_cxx)?;
        Ok(Column { inner: result })
    }
}
