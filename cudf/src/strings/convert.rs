//! String-to-numeric and numeric-to-string conversion operations.

use crate::column::Column;
use crate::error::{CudfError, Result};
use crate::types::DataType;

impl Column {
    /// Convert a string column to an integer column of the specified type.
    pub fn str_to_integers(&self, dtype: DataType) -> Result<Column> {
        let result = cudf_cxx::strings::convert::ffi::str_to_integers(
            &self.inner,
            dtype.id() as i32,
        )
        .map_err(CudfError::from_cxx)?;
        Ok(Column { inner: result })
    }

    /// Convert an integer column to a string column.
    pub fn str_from_integers(&self) -> Result<Column> {
        let result =
            cudf_cxx::strings::convert::ffi::str_from_integers(&self.inner)
                .map_err(CudfError::from_cxx)?;
        Ok(Column { inner: result })
    }

    /// Convert a string column to a float column of the specified type.
    pub fn str_to_floats(&self, dtype: DataType) -> Result<Column> {
        let result = cudf_cxx::strings::convert::ffi::str_to_floats(
            &self.inner,
            dtype.id() as i32,
        )
        .map_err(CudfError::from_cxx)?;
        Ok(Column { inner: result })
    }

    /// Convert a float column to a string column.
    pub fn str_from_floats(&self) -> Result<Column> {
        let result =
            cudf_cxx::strings::convert::ffi::str_from_floats(&self.inner)
                .map_err(CudfError::from_cxx)?;
        Ok(Column { inner: result })
    }
}
