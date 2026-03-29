//! String translate and character filter operations.

use crate::column::Column;
use crate::error::{CudfError, Result};

impl Column {
    /// Translate characters using parallel arrays of source/target code points.
    ///
    /// For each character in the source array, the corresponding character
    /// in the target array is substituted. A target of 0 removes the character.
    pub fn str_translate(&self, src_chars: &[u32], dst_chars: &[u32]) -> Result<Column> {
        let result =
            cudf_cxx::strings::translate::ffi::str_translate(&self.inner, src_chars, dst_chars)
                .map_err(CudfError::from_cxx)?;
        Ok(Column { inner: result })
    }

    /// Filter characters by keeping or removing specified code point ranges.
    ///
    /// `range_pairs` contains consecutive (lo, hi) pairs. If `keep` is true,
    /// only characters within the ranges are kept; otherwise they are removed.
    pub fn str_filter_characters(
        &self,
        range_pairs: &[u32],
        keep: bool,
        replacement: &str,
    ) -> Result<Column> {
        let result = cudf_cxx::strings::translate::ffi::str_filter_characters(
            &self.inner,
            range_pairs,
            keep,
            replacement,
        )
        .map_err(CudfError::from_cxx)?;
        Ok(Column { inner: result })
    }
}
