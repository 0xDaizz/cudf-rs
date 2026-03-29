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
        let raw =
            cudf_cxx::null_mask::ffi::set_null_mask_from_host(&self.inner, mask, null_count as i32)
                .map_err(CudfError::from_cxx)?;
        Ok(Column { inner: raw })
    }

    /// Set a range of bits in this column's null mask.
    ///
    /// Returns a new column with bits `[begin, end)` set to `valid`.
    /// If `valid` is `true`, elements in the range are marked as valid;
    /// if `false`, they are marked as null.
    pub fn set_null_mask_range(&self, begin: usize, end: usize, valid: bool) -> Result<Column> {
        let raw = cudf_cxx::null_mask::ffi::set_null_mask_range(
            &self.inner,
            begin as i32,
            end as i32,
            valid,
        )
        .map_err(CudfError::from_cxx)?;
        Ok(Column { inner: raw })
    }
}

/// Copy a column's bitmask to host bytes.
///
/// Returns an empty vector if the column has no null mask (all valid).
pub fn copy_bitmask(col: &Column) -> Vec<u8> {
    cudf_cxx::null_mask::ffi::copy_bitmask_to_host(&col.inner)
}

/// Result of a bitmask AND or OR operation.
pub struct BitmaskResult {
    /// The combined bitmask bytes.
    pub mask: Vec<u8>,
    /// The number of null (unset) bits.
    pub null_count: usize,
}

/// Compute bitwise AND of null masks from multiple columns.
///
/// A bit is set in the output only if it is set in ALL input columns.
/// Use this to find rows that are valid across all columns.
pub fn bitmask_and(columns: &[&Column]) -> Result<BitmaskResult> {
    let mut builder = cudf_cxx::null_mask::ffi::bitmask_builder_new();
    for col in columns {
        builder.pin_mut().add_column(&col.inner);
    }
    let result = cudf_cxx::null_mask::ffi::bitmask_and(&builder).map_err(CudfError::from_cxx)?;
    Ok(BitmaskResult {
        mask: result.get_mask(),
        null_count: result.get_null_count() as usize,
    })
}

/// Compute the null count for a given mask state and size.
///
/// For example, `state_null_count(MaskState::AllNull, 100)` returns 100.
pub fn state_null_count(state: MaskState, size: usize) -> Result<usize> {
    let count = cudf_cxx::null_mask::ffi::state_null_count(state as i32, size as i32)
        .map_err(CudfError::from_cxx)?;
    Ok(count as usize)
}

/// Compute the number of `bitmask_type` words needed for the given number of bits.
///
/// This is the actual number of words, not the padded allocation size.
pub fn num_bitmask_words(num_bits: usize) -> usize {
    cudf_cxx::null_mask::ffi::num_bitmask_words(num_bits as i32) as usize
}

/// Compute bitwise OR of null masks from multiple columns.
///
/// A bit is set in the output if it is set in ANY input column.
/// Use this to find rows that are valid in at least one column.
pub fn bitmask_or(columns: &[&Column]) -> Result<BitmaskResult> {
    let mut builder = cudf_cxx::null_mask::ffi::bitmask_builder_new();
    for col in columns {
        builder.pin_mut().add_column(&col.inner);
    }
    let result = cudf_cxx::null_mask::ffi::bitmask_or(&builder).map_err(CudfError::from_cxx)?;
    Ok(BitmaskResult {
        mask: result.get_mask(),
        null_count: result.get_null_count() as usize,
    })
}
