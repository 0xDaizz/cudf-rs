//! String partition operations.

use crate::column::Column;
use crate::error::{CudfError, Result};
use crate::table::Table;

impl Column {
    /// Partition each string at the first occurrence of `delimiter`.
    ///
    /// Returns a 3-column table: `[before, delimiter, after]`.
    pub fn str_partition(&self, delimiter: &str) -> Result<Table> {
        let result = cudf_cxx::strings::partition::ffi::str_partition(&self.inner, delimiter)
            .map_err(CudfError::from_cxx)?;
        Ok(Table { inner: result })
    }

    /// Partition each string at the last occurrence of `delimiter`.
    ///
    /// Returns a 3-column table: `[before, delimiter, after]`.
    pub fn str_rpartition(&self, delimiter: &str) -> Result<Table> {
        let result = cudf_cxx::strings::partition::ffi::str_rpartition(&self.inner, delimiter)
            .map_err(CudfError::from_cxx)?;
        Ok(Table { inner: result })
    }
}
