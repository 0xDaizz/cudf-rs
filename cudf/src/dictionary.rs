//! GPU-accelerated dictionary encoding and decoding.
//!
//! Dictionary encoding replaces repeated values with integer indices into
//! a sorted key table, which can significantly reduce memory usage and
//! improve performance for columns with low cardinality.
//!
//! # Examples
//!
//! ```rust,no_run
//! use cudf::Column;
//!
//! let col = Column::from_slice(&[1i32, 2, 1, 2, 1]).unwrap();
//! let encoded = col.dictionary_encode().unwrap();
//! let decoded = encoded.dictionary_decode().unwrap();
//! ```

use crate::column::Column;
use crate::error::{CudfError, Result};
use crate::scalar::Scalar;
use crate::table::Table;

impl Column {
    /// Dictionary-encode this column.
    ///
    /// Returns a DICTIONARY type column with sorted unique keys and
    /// integer indices. Null values are preserved in the output.
    ///
    /// # Errors
    ///
    /// Returns an error if the column is already dictionary-encoded
    /// or a GPU error occurs.
    pub fn dictionary_encode(&self) -> Result<Column> {
        let raw = cudf_cxx::dictionary::ffi::dictionary_encode(&self.inner)
            .map_err(CudfError::from_cxx)?;
        Ok(Column { inner: raw })
    }

    /// Decode a dictionary-encoded column back to its original type.
    ///
    /// Gathers keys using the indices to reconstruct the original values.
    ///
    /// # Errors
    ///
    /// Returns an error if this column is not dictionary-encoded
    /// or a GPU error occurs.
    pub fn dictionary_decode(&self) -> Result<Column> {
        let raw = cudf_cxx::dictionary::ffi::dictionary_decode(&self.inner)
            .map_err(CudfError::from_cxx)?;
        Ok(Column { inner: raw })
    }

    /// Get the index of a key in a dictionary column.
    ///
    /// Returns a scalar containing the index, or an invalid scalar
    /// if the key is not found.
    ///
    /// # Errors
    ///
    /// Returns an error if this column is not dictionary-encoded.
    pub fn dictionary_get_index(&self, key: &Scalar) -> Result<Scalar> {
        let raw = cudf_cxx::dictionary::ffi::dictionary_get_index(&self.inner, &key.inner)
            .map_err(CudfError::from_cxx)?;
        Ok(Scalar { inner: raw })
    }

    /// Add new keys to a dictionary column.
    ///
    /// Returns a new dictionary column with the new keys added.
    /// Existing indices are remapped to the updated key set.
    ///
    /// # Errors
    ///
    /// Returns an error if this column is not dictionary-encoded.
    pub fn dictionary_add_keys(&self, new_keys: &Column) -> Result<Column> {
        let raw = cudf_cxx::dictionary::ffi::dictionary_add_keys(&self.inner, &new_keys.inner)
            .map_err(CudfError::from_cxx)?;
        Ok(Column { inner: raw })
    }

    /// Remove specified keys from a dictionary column.
    ///
    /// Rows that refer to removed keys become null.
    ///
    /// # Errors
    ///
    /// Returns an error if this column is not dictionary-encoded.
    pub fn dictionary_remove_keys(&self, keys_to_remove: &Column) -> Result<Column> {
        let raw =
            cudf_cxx::dictionary::ffi::dictionary_remove_keys(&self.inner, &keys_to_remove.inner)
                .map_err(CudfError::from_cxx)?;
        Ok(Column { inner: raw })
    }

    /// Remove unused keys from a dictionary column.
    ///
    /// Keys not referenced by any index are removed, and indices are
    /// remapped accordingly.
    ///
    /// # Errors
    ///
    /// Returns an error if this column is not dictionary-encoded.
    pub fn dictionary_remove_unused_keys(&self) -> Result<Column> {
        let raw = cudf_cxx::dictionary::ffi::dictionary_remove_unused_keys(&self.inner)
            .map_err(CudfError::from_cxx)?;
        Ok(Column { inner: raw })
    }

    /// Replace all keys in a dictionary column with new keys.
    ///
    /// Indices are remapped to match the new key set. Rows whose
    /// values are not in `new_keys` become null.
    ///
    /// # Errors
    ///
    /// Returns an error if this column is not dictionary-encoded.
    pub fn dictionary_set_keys(&self, new_keys: &Column) -> Result<Column> {
        let raw = cudf_cxx::dictionary::ffi::dictionary_set_keys(&self.inner, &new_keys.inner)
            .map_err(CudfError::from_cxx)?;
        Ok(Column { inner: raw })
    }
}

/// Match dictionaries across multiple dictionary columns.
///
/// After matching, all columns share the same key set, enabling
/// operations like join or concatenation across dictionary columns
/// with different key sets.
///
/// Returns a table containing the matched dictionary columns.
pub fn dictionary_match_dictionaries(columns: Vec<Column>) -> Result<Table> {
    let mut builder = cudf_cxx::dictionary::ffi::dictionary_match_builder_new();
    for col in columns {
        builder.pin_mut().add_column(col.inner);
    }
    let raw = cudf_cxx::dictionary::ffi::dictionary_match_dictionaries(builder)
        .map_err(CudfError::from_cxx)?;
    Ok(Table { inner: raw })
}
