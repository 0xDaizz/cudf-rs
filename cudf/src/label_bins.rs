//! GPU-accelerated bin labeling.
//!
//! Labels elements based on their membership in specified bins.

use crate::column::Column;
use crate::error::{CudfError, Result};

impl Column {
    /// Label elements based on membership in the specified bins.
    ///
    /// A bin `i` is defined by `[left_edges[i], right_edges[i]]` with
    /// inclusiveness controlled by `left_inclusive` and `right_inclusive`.
    ///
    /// Returns an integer column of bin labels. Elements that do not
    /// belong to any bin are labeled as null.
    ///
    /// # Arguments
    ///
    /// * `left_edges` - Column of left bin edges (must be monotonically increasing)
    /// * `left_inclusive` - Whether left edges are inclusive
    /// * `right_edges` - Column of right bin edges
    /// * `right_inclusive` - Whether right edges are inclusive
    pub fn label_bins(
        &self,
        left_edges: &Column,
        left_inclusive: bool,
        right_edges: &Column,
        right_inclusive: bool,
    ) -> Result<Column> {
        let result = cudf_cxx::label_bins::ffi::label_bins(
            &self.inner,
            &left_edges.inner,
            left_inclusive,
            &right_edges.inner,
            right_inclusive,
        )
        .map_err(CudfError::from_cxx)?;
        Ok(Column { inner: result })
    }
}
