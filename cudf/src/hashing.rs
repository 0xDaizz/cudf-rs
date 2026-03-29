//! GPU-accelerated hashing operations.
//!
//! Provides row-wise hashing of [`Table`]s using various algorithms.
//!
//! # Examples
//!
//! ```rust,no_run
//! use cudf::{Column, Table};
//!
//! let col = Column::from_slice(&[1i32, 2, 3]).unwrap();
//! let table = Table::new(vec![col]).unwrap();
//! let hashes = table.hash_murmur3(0).unwrap();
//! assert_eq!(hashes.len(), 3);
//! ```

use crate::column::Column;
use crate::error::{CudfError, Result};
use crate::table::Table;

impl Table {
    /// Hash each row using MurmurHash3 (32-bit).
    ///
    /// Returns an `i32` column of hash values.
    pub fn hash_murmur3(&self, seed: u32) -> Result<Column> {
        let raw =
            cudf_cxx::hashing::ffi::hash_murmur3(&self.inner, seed).map_err(CudfError::from_cxx)?;
        Ok(Column { inner: raw })
    }

    /// Hash each row using xxHash64.
    ///
    /// Returns an `i64` column of hash values.
    pub fn hash_xxhash64(&self, seed: u64) -> Result<Column> {
        let raw = cudf_cxx::hashing::ffi::hash_xxhash64(&self.inner, seed)
            .map_err(CudfError::from_cxx)?;
        Ok(Column { inner: raw })
    }

    /// Hash each row using MD5.
    ///
    /// Returns a string column of hex-encoded hash values.
    pub fn hash_md5(&self) -> Result<Column> {
        let raw = cudf_cxx::hashing::ffi::hash_md5(&self.inner).map_err(CudfError::from_cxx)?;
        Ok(Column { inner: raw })
    }

    /// Hash each row using SHA-256.
    ///
    /// Returns a string column of hex-encoded hash values.
    pub fn hash_sha256(&self) -> Result<Column> {
        let raw = cudf_cxx::hashing::ffi::hash_sha256(&self.inner).map_err(CudfError::from_cxx)?;
        Ok(Column { inner: raw })
    }

    /// Hash each row using SHA-1.
    ///
    /// Returns a string column of hex-encoded hash values.
    pub fn hash_sha1(&self) -> Result<Column> {
        let raw = cudf_cxx::hashing::ffi::hash_sha1(&self.inner).map_err(CudfError::from_cxx)?;
        Ok(Column { inner: raw })
    }

    /// Hash each row using SHA-224.
    ///
    /// Returns a string column of hex-encoded hash values.
    pub fn hash_sha224(&self) -> Result<Column> {
        let raw = cudf_cxx::hashing::ffi::hash_sha224(&self.inner).map_err(CudfError::from_cxx)?;
        Ok(Column { inner: raw })
    }

    /// Hash each row using SHA-384.
    ///
    /// Returns a string column of hex-encoded hash values.
    pub fn hash_sha384(&self) -> Result<Column> {
        let raw = cudf_cxx::hashing::ffi::hash_sha384(&self.inner).map_err(CudfError::from_cxx)?;
        Ok(Column { inner: raw })
    }

    /// Hash each row using SHA-512.
    ///
    /// Returns a string column of hex-encoded hash values.
    pub fn hash_sha512(&self) -> Result<Column> {
        let raw = cudf_cxx::hashing::ffi::hash_sha512(&self.inner).map_err(CudfError::from_cxx)?;
        Ok(Column { inner: raw })
    }

    /// Hash each row using xxHash32.
    ///
    /// Returns an `i32` column of hash values.
    pub fn hash_xxhash32(&self, seed: u32) -> Result<Column> {
        let raw = cudf_cxx::hashing::ffi::hash_xxhash32(&self.inner, seed)
            .map_err(CudfError::from_cxx)?;
        Ok(Column { inner: raw })
    }

    /// Hash each row using MurmurHash3 x64 128-bit.
    ///
    /// Returns a 2-column table where the two columns together form
    /// a 128-bit hash per row.
    pub fn hash_murmurhash3_x64_128(&self, seed: u64) -> Result<Table> {
        let raw = cudf_cxx::hashing::ffi::hash_murmurhash3_x64_128(&self.inner, seed)
            .map_err(CudfError::from_cxx)?;
        Ok(Table { inner: raw })
    }
}
