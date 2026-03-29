//! String find/search operations.

use crate::column::Column;
use crate::error::{CudfError, Result};
use crate::table::Table;

impl Column {
    /// Find first occurrence of `target` in each string, starting at `start`.
    ///
    /// Returns an INT32 column of positions (-1 if not found).
    pub fn str_find(&self, target: &str, start: usize) -> Result<Column> {
        let result = cudf_cxx::strings::find::ffi::str_find(&self.inner, target, start as i32)
            .map_err(CudfError::from_cxx)?;
        Ok(Column { inner: result })
    }

    /// Find last occurrence of `target` in each string.
    ///
    /// Returns an INT32 column of positions (-1 if not found).
    pub fn str_rfind(&self, target: &str) -> Result<Column> {
        let result = cudf_cxx::strings::find::ffi::str_rfind(&self.inner, target)
            .map_err(CudfError::from_cxx)?;
        Ok(Column { inner: result })
    }

    /// Check if each string starts with `target`.
    ///
    /// Returns a BOOL8 column.
    pub fn str_starts_with(&self, target: &str) -> Result<Column> {
        let result = cudf_cxx::strings::find::ffi::str_starts_with(&self.inner, target)
            .map_err(CudfError::from_cxx)?;
        Ok(Column { inner: result })
    }

    /// Check if each string ends with `target`.
    ///
    /// Returns a BOOL8 column.
    pub fn str_ends_with(&self, target: &str) -> Result<Column> {
        let result = cudf_cxx::strings::find::ffi::str_ends_with(&self.inner, target)
            .map_err(CudfError::from_cxx)?;
        Ok(Column { inner: result })
    }

    /// Check if each string contains any of the target strings.
    ///
    /// Returns a table of BOOL8 columns, one per target.
    pub fn str_contains_multiple(&self, targets: &Column) -> Result<Table> {
        let result =
            cudf_cxx::strings::find::ffi::str_contains_multiple(&self.inner, &targets.inner)
                .map_err(CudfError::from_cxx)?;
        Ok(Table { inner: result })
    }

    /// Find positions of multiple target strings in each input string.
    ///
    /// Returns a lists column of INT32 positions. Each row contains
    /// one position per target string (-1 if not found).
    pub fn str_find_multiple(&self, targets: &Column) -> Result<Column> {
        let result = cudf_cxx::strings::find::ffi::str_find_multiple(&self.inner, &targets.inner)
            .map_err(CudfError::from_cxx)?;
        Ok(Column { inner: result })
    }

    /// Find each target (from a column) in the corresponding row string.
    ///
    /// Returns an INT32 column of positions (-1 if not found).
    pub fn str_find_column(&self, targets: &Column, start: usize) -> Result<Column> {
        let result = cudf_cxx::strings::find::ffi::str_find_column(
            &self.inner,
            &targets.inner,
            start as i32,
        )
        .map_err(CudfError::from_cxx)?;
        Ok(Column { inner: result })
    }

    /// Find the nth instance of `target` in each string.
    ///
    /// Returns an INT32 column of positions (-1 if not found).
    pub fn str_find_instance(&self, target: &str, instance: usize) -> Result<Column> {
        let result =
            cudf_cxx::strings::find::ffi::str_find_instance(&self.inner, target, instance as i32)
                .map_err(CudfError::from_cxx)?;
        Ok(Column { inner: result })
    }

    /// Check if each string contains the corresponding target (column-based).
    ///
    /// Returns a BOOL8 column.
    pub fn str_contains_column(&self, targets: &Column) -> Result<Column> {
        let result = cudf_cxx::strings::find::ffi::str_contains_column(&self.inner, &targets.inner)
            .map_err(CudfError::from_cxx)?;
        Ok(Column { inner: result })
    }

    /// Check if each string starts with the corresponding target (column-based).
    ///
    /// Returns a BOOL8 column.
    pub fn str_starts_with_column(&self, targets: &Column) -> Result<Column> {
        let result =
            cudf_cxx::strings::find::ffi::str_starts_with_column(&self.inner, &targets.inner)
                .map_err(CudfError::from_cxx)?;
        Ok(Column { inner: result })
    }

    /// Check if each string ends with the corresponding target (column-based).
    ///
    /// Returns a BOOL8 column.
    pub fn str_ends_with_column(&self, targets: &Column) -> Result<Column> {
        let result =
            cudf_cxx::strings::find::ffi::str_ends_with_column(&self.inner, &targets.inner)
                .map_err(CudfError::from_cxx)?;
        Ok(Column { inner: result })
    }
}
