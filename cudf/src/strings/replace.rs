//! String replace operations.

use crate::column::Column;
use crate::error::{CudfError, Result};

impl Column {
    /// Replace all occurrences of `target` with `replacement` in each string.
    pub fn str_replace(&self, target: &str, replacement: &str) -> Result<Column> {
        let result = cudf_cxx::strings::replace::ffi::str_replace(&self.inner, target, replacement)
            .map_err(CudfError::from_cxx)?;
        Ok(Column { inner: result })
    }

    /// Replace all regex matches of `pattern` with `replacement` in each string.
    pub fn str_replace_re(&self, pattern: &str, replacement: &str) -> Result<Column> {
        let result =
            cudf_cxx::strings::replace::ffi::str_replace_re(&self.inner, pattern, replacement)
                .map_err(CudfError::from_cxx)?;
        Ok(Column { inner: result })
    }

    /// Replace characters in the `[start, stop)` range with `replacement` in each string.
    ///
    /// Use `stop = -1` to replace through end of string.
    pub fn str_replace_slice(&self, replacement: &str, start: i32, stop: i32) -> Result<Column> {
        let result = cudf_cxx::strings::replace::ffi::str_replace_slice(
            &self.inner,
            replacement,
            start,
            stop,
        )
        .map_err(CudfError::from_cxx)?;
        Ok(Column { inner: result })
    }

    /// Replace multiple target strings with corresponding replacement strings.
    ///
    /// Both `targets` and `replacements` must be string columns. All occurrences
    /// of each target are replaced in every input string.
    pub fn str_replace_multiple(&self, targets: &Column, replacements: &Column) -> Result<Column> {
        let result = cudf_cxx::strings::replace::ffi::str_replace_multiple(
            &self.inner,
            &targets.inner,
            &replacements.inner,
        )
        .map_err(CudfError::from_cxx)?;
        Ok(Column { inner: result })
    }

    /// Replace regex matches using back-references in the replacement template.
    ///
    /// The `replacement` template can reference capture groups via `\1`, `\2`, etc.
    pub fn str_replace_with_backrefs(&self, pattern: &str, replacement: &str) -> Result<Column> {
        let result = cudf_cxx::strings::replace::ffi::str_replace_with_backrefs(
            &self.inner,
            pattern,
            replacement,
        )
        .map_err(CudfError::from_cxx)?;
        Ok(Column { inner: result })
    }

    /// Replace multiple regex patterns with corresponding replacements from a column.
    ///
    /// Each `patterns[i]` is applied in sequence, and matched text is replaced
    /// with `replacements[i]`. The `replacements` column must be a string column.
    pub fn str_replace_re_multiple(
        &self,
        patterns: &[String],
        replacements: &Column,
    ) -> Result<Column> {
        let result = cudf_cxx::strings::replace::ffi::str_replace_re_multiple(
            &self.inner,
            patterns,
            &replacements.inner,
        )
        .map_err(CudfError::from_cxx)?;
        Ok(Column { inner: result })
    }
}
