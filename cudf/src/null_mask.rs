//! Null mask utilities for GPU columns.
//!
//! Provides functions for inspecting and manipulating the validity
//! bitmask of columns. Each bit in the mask indicates whether the
//! corresponding element is valid (1) or null (0).
//!
//! # Examples
//!
//! ```rust,no_run
//! use cudf::Column;
//! use cudf::null_mask;
//!
//! let col = Column::from_slice(&[1i32, 2, 3]).unwrap();
//! let mask = null_mask::null_mask_to_host(&col).unwrap();
//! ```

use crate::column::Column;
use crate::error::{CudfError, Result};

/// State for creating null masks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MaskState {
    /// No mask is allocated.
    Unallocated = 0,
    /// Mask is allocated but not initialized.
    Uninitialized = 1,
    /// All elements are valid (no nulls).
    AllValid = 2,
    /// All elements are null.
    AllNull = 3,
}

/// Compute the number of bytes needed to store a bitmask for `num_bits` elements.
pub fn bitmask_allocation_size(num_bits: usize) -> usize {
    cudf_cxx::null_mask::ffi::bitmask_allocation_size(num_bits as i32) as usize
}

/// Copy a column's null mask to a host byte vector.
///
/// Each bit indicates whether the corresponding element is valid (1) or null (0).
/// If the column has no null mask, returns a vector with all bits set to 1.
pub fn null_mask_to_host(col: &Column) -> Result<Vec<u8>> {
    let num_bytes = bitmask_allocation_size(col.len());
    if num_bytes == 0 {
        return Ok(Vec::new());
    }
    let mut buf = vec![0u8; num_bytes];
    cudf_cxx::null_mask::ffi::copy_null_mask_to_host(&col.inner, &mut buf)
        .map_err(CudfError::from_cxx)?;
    Ok(buf)
}

/// Count the number of null values in a column.
pub fn null_count(col: &Column) -> usize {
    cudf_cxx::null_mask::ffi::null_count_column(&col.inner) as usize
}

impl Column {
    /// Create a new column with a null mask applied from host-side bytes.
    ///
    /// The `mask` byte slice is a bitmask where each bit indicates whether
    /// the corresponding element is valid (1) or null (0). `null_count`
    /// must match the number of unset bits in the mask.
    ///
    /// # Errors
    ///
    /// Returns an error if the mask is too small or a GPU error occurs.
    pub fn with_null_mask(&self, mask: &[u8], null_count: usize) -> Result<Column> {
        let raw = cudf_cxx::null_mask::ffi::set_null_mask_from_host(
            &self.inner,
            mask,
            null_count as i32,
        )
        .map_err(CudfError::from_cxx)?;
        Ok(Column { inner: raw })
    }
}
