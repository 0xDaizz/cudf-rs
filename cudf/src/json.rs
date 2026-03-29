//! GPU-accelerated JSONPath operations.
//!
//! Apply JSONPath queries to columns of JSON strings.

use crate::column::Column;
use crate::error::{CudfError, Result};

/// Options for controlling `get_json_object` behavior.
#[derive(Debug, Clone)]
pub struct JsonObjectOptions {
    /// Allow single quotes to represent strings in JSON.
    pub allow_single_quotes: bool,
    /// Strip quotes from individually returned string values.
    pub strip_quotes: bool,
    /// Return nulls when an object does not contain the requested field.
    pub missing_fields_as_nulls: bool,
}

impl Default for JsonObjectOptions {
    fn default() -> Self {
        Self {
            allow_single_quotes: false,
            strip_quotes: true,
            missing_fields_as_nulls: false,
        }
    }
}

impl Column {
    /// Apply a JSONPath query to each JSON string in the column.
    ///
    /// Implements `$`, `.`, `[]`, and `*` operators from JSONPath.
    ///
    /// # Arguments
    ///
    /// * `json_path` - JSONPath expression (e.g., `$.store.book[0].title`)
    /// * `options` - Options controlling quote handling and null behavior
    pub fn get_json_object(&self, json_path: &str, options: &JsonObjectOptions) -> Result<Column> {
        let result = cudf_cxx::json::ffi::get_json_object(
            &self.inner,
            json_path,
            options.allow_single_quotes,
            options.strip_quotes,
            options.missing_fields_as_nulls,
        )
        .map_err(CudfError::from_cxx)?;
        Ok(Column { inner: result })
    }
}
