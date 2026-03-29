//! String-to-numeric and numeric-to-string conversion operations.

use crate::column::Column;
use crate::error::{CudfError, Result};
use crate::types::DataType;

impl Column {
    /// Convert a string column to an integer column of the specified type.
    pub fn str_to_integers(&self, dtype: DataType) -> Result<Column> {
        let result =
            cudf_cxx::strings::convert::ffi::str_to_integers(&self.inner, dtype.id() as i32)
                .map_err(CudfError::from_cxx)?;
        Ok(Column { inner: result })
    }

    /// Convert an integer column to a string column.
    pub fn str_from_integers(&self) -> Result<Column> {
        let result = cudf_cxx::strings::convert::ffi::str_from_integers(&self.inner)
            .map_err(CudfError::from_cxx)?;
        Ok(Column { inner: result })
    }

    /// Convert a string column to a float column of the specified type.
    pub fn str_to_floats(&self, dtype: DataType) -> Result<Column> {
        let result = cudf_cxx::strings::convert::ffi::str_to_floats(&self.inner, dtype.id() as i32)
            .map_err(CudfError::from_cxx)?;
        Ok(Column { inner: result })
    }

    /// Convert a float column to a string column.
    pub fn str_from_floats(&self) -> Result<Column> {
        let result = cudf_cxx::strings::convert::ffi::str_from_floats(&self.inner)
            .map_err(CudfError::from_cxx)?;
        Ok(Column { inner: result })
    }

    // ── Booleans ──────────────────────────────────────────────

    /// Convert a string column to a boolean column.
    ///
    /// Strings matching `true_str` produce true; all others produce false.
    pub fn str_to_booleans(&self, true_str: &str) -> Result<Column> {
        let result = cudf_cxx::strings::convert::ffi::str_to_booleans(&self.inner, true_str)
            .map_err(CudfError::from_cxx)?;
        Ok(Column { inner: result })
    }

    /// Convert a boolean column to a string column.
    pub fn str_from_booleans(&self, true_str: &str, false_str: &str) -> Result<Column> {
        let result =
            cudf_cxx::strings::convert::ffi::str_from_booleans(&self.inner, true_str, false_str)
                .map_err(CudfError::from_cxx)?;
        Ok(Column { inner: result })
    }

    // ── Timestamps ────────────────────────────────────────────

    /// Convert a string column to a timestamp column.
    pub fn str_to_timestamps(&self, format: &str, dtype: DataType) -> Result<Column> {
        let result = cudf_cxx::strings::convert::ffi::str_to_timestamps(
            &self.inner,
            format,
            dtype.id() as i32,
        )
        .map_err(CudfError::from_cxx)?;
        Ok(Column { inner: result })
    }

    /// Convert a timestamp column to a string column.
    pub fn str_from_timestamps(&self, format: &str) -> Result<Column> {
        let result = cudf_cxx::strings::convert::ffi::str_from_timestamps(&self.inner, format)
            .map_err(CudfError::from_cxx)?;
        Ok(Column { inner: result })
    }

    // ── Durations ─────────────────────────────────────────────

    /// Convert a string column to a duration column.
    pub fn str_to_durations(&self, format: &str, dtype: DataType) -> Result<Column> {
        let result = cudf_cxx::strings::convert::ffi::str_to_durations(
            &self.inner,
            format,
            dtype.id() as i32,
        )
        .map_err(CudfError::from_cxx)?;
        Ok(Column { inner: result })
    }

    /// Convert a duration column to a string column.
    pub fn str_from_durations(&self, format: &str) -> Result<Column> {
        let result = cudf_cxx::strings::convert::ffi::str_from_durations(&self.inner, format)
            .map_err(CudfError::from_cxx)?;
        Ok(Column { inner: result })
    }

    // ── Fixed Point ───────────────────────────────────────────

    /// Convert a string column to a fixed-point (decimal) column.
    pub fn str_to_fixed_point(&self, dtype: DataType, scale: i32) -> Result<Column> {
        let result = cudf_cxx::strings::convert::ffi::str_to_fixed_point(
            &self.inner,
            dtype.id() as i32,
            scale,
        )
        .map_err(CudfError::from_cxx)?;
        Ok(Column { inner: result })
    }

    /// Convert a fixed-point column to a string column.
    pub fn str_from_fixed_point(&self) -> Result<Column> {
        let result = cudf_cxx::strings::convert::ffi::str_from_fixed_point(&self.inner)
            .map_err(CudfError::from_cxx)?;
        Ok(Column { inner: result })
    }

    // ── Type Checks ───────────────────────────────────────────

    /// Check if each string is a valid integer representation.
    pub fn str_is_integer(&self) -> Result<Column> {
        let result = cudf_cxx::strings::convert::ffi::str_is_integer(&self.inner)
            .map_err(CudfError::from_cxx)?;
        Ok(Column { inner: result })
    }

    /// Check if each string is a valid float representation.
    pub fn str_is_float(&self) -> Result<Column> {
        let result = cudf_cxx::strings::convert::ffi::str_is_float(&self.inner)
            .map_err(CudfError::from_cxx)?;
        Ok(Column { inner: result })
    }

    // ── Hex ───────────────────────────────────────────────────

    /// Convert hex string column to integer column.
    pub fn str_hex_to_integers(&self, dtype: DataType) -> Result<Column> {
        let result =
            cudf_cxx::strings::convert::ffi::str_hex_to_integers(&self.inner, dtype.id() as i32)
                .map_err(CudfError::from_cxx)?;
        Ok(Column { inner: result })
    }

    /// Convert integer column to hex string column.
    pub fn str_integers_to_hex(&self) -> Result<Column> {
        let result = cudf_cxx::strings::convert::ffi::str_integers_to_hex(&self.inner)
            .map_err(CudfError::from_cxx)?;
        Ok(Column { inner: result })
    }

    // ── IPv4 ──────────────────────────────────────────────────

    /// Convert IPv4 string column to integer column.
    pub fn str_ipv4_to_integers(&self) -> Result<Column> {
        let result = cudf_cxx::strings::convert::ffi::str_ipv4_to_integers(&self.inner)
            .map_err(CudfError::from_cxx)?;
        Ok(Column { inner: result })
    }

    /// Convert integer column to IPv4 string column.
    pub fn str_integers_to_ipv4(&self) -> Result<Column> {
        let result = cudf_cxx::strings::convert::ffi::str_integers_to_ipv4(&self.inner)
            .map_err(CudfError::from_cxx)?;
        Ok(Column { inner: result })
    }

    // ── URL ───────────────────────────────────────────────────

    /// URL-encode each string.
    pub fn str_url_encode(&self) -> Result<Column> {
        let result = cudf_cxx::strings::convert::ffi::str_url_encode(&self.inner)
            .map_err(CudfError::from_cxx)?;
        Ok(Column { inner: result })
    }

    /// URL-decode each string.
    pub fn str_url_decode(&self) -> Result<Column> {
        let result = cudf_cxx::strings::convert::ffi::str_url_decode(&self.inner)
            .map_err(CudfError::from_cxx)?;
        Ok(Column { inner: result })
    }
}
