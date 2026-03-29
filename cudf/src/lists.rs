//! GPU-accelerated list column operations.
//!
//! Provides operations on list (nested) columns: exploding lists into rows,
//! sorting elements within lists, checking containment, and extracting elements.
//!
//! # Examples
//!
//! ```rust,no_run
//! use cudf::Column;
//!
//! // Extract the first element from each list row
//! let list_col: Column = todo!("create a list column");
//! let first_elements = list_col.lists_extract(0).unwrap();
//! ```

use crate::column::Column;
use crate::error::{CudfError, Result};
use crate::scalar::Scalar;
use crate::sorting::NullOrder;
use crate::table::Table;

impl Table {
    /// Explode a list column, expanding each list element into its own row.
    ///
    /// The column at `explode_col_idx` must be a list column. Each element
    /// of the list becomes a separate row in the output table. Null lists
    /// are dropped.
    ///
    /// # Errors
    ///
    /// Returns an error if the column index is out of bounds, the column is
    /// not a list type, or a GPU error occurs.
    pub fn lists_explode(&self, explode_col_idx: usize) -> Result<Table> {
        let raw = cudf_cxx::lists::ops::ffi::lists_explode(&self.inner, explode_col_idx as i32)
            .map_err(CudfError::from_cxx)?;
        Ok(Table { inner: raw })
    }

    /// Explode a list column, retaining null entries and empty lists as null rows.
    ///
    /// Like [`lists_explode`](Self::lists_explode) but null and empty lists
    /// produce a null row instead of being dropped.
    pub fn lists_explode_outer(&self, explode_col_idx: usize) -> Result<Table> {
        let raw =
            cudf_cxx::lists::ops::ffi::lists_explode_outer(&self.inner, explode_col_idx as i32)
                .map_err(CudfError::from_cxx)?;
        Ok(Table { inner: raw })
    }
}

impl Column {
    /// Sort elements within each list row.
    ///
    /// Returns a new list column where the elements within each row are
    /// sorted according to the given order and null placement.
    pub fn lists_sort(&self, ascending: bool, null_order: NullOrder) -> Result<Column> {
        let raw = cudf_cxx::lists::ops::ffi::lists_sort(&self.inner, ascending, null_order as i32)
            .map_err(CudfError::from_cxx)?;
        Ok(Column { inner: raw })
    }

    /// Check whether each list row contains the given scalar value.
    ///
    /// Returns a boolean column where `true` indicates the list row
    /// contains the search key.
    pub fn lists_contains(&self, search_key: &Scalar) -> Result<Column> {
        let raw = cudf_cxx::lists::ops::ffi::lists_contains(&self.inner, &search_key.inner)
            .map_err(CudfError::from_cxx)?;
        Ok(Column { inner: raw })
    }

    /// Check whether each list row contains any null elements.
    ///
    /// Returns a boolean column where `true` indicates the list row
    /// has at least one null element.
    pub fn lists_contains_nulls(&self) -> Result<Column> {
        let raw = cudf_cxx::lists::ops::ffi::lists_contains_nulls(&self.inner)
            .map_err(CudfError::from_cxx)?;
        Ok(Column { inner: raw })
    }

    /// Extract the element at `index` from each list row.
    ///
    /// Negative indices count from the end of each list. If `index` is
    /// out of bounds for a given row, the output for that row is null.
    pub fn lists_extract(&self, index: i32) -> Result<Column> {
        let raw = cudf_cxx::lists::ops::ffi::lists_extract(&self.inner, index)
            .map_err(CudfError::from_cxx)?;
        Ok(Column { inner: raw })
    }

    /// Count the number of elements in each list row.
    ///
    /// Returns an INT32 column of counts.
    pub fn lists_count_elements(&self) -> Result<Column> {
        let raw = cudf_cxx::lists::ops::ffi::lists_count_elements(&self.inner)
            .map_err(CudfError::from_cxx)?;
        Ok(Column { inner: raw })
    }

    /// Find the first position of `key` in each list row.
    ///
    /// Returns an INT32 column where -1 means not found.
    pub fn lists_index_of_scalar(&self, key: &Scalar) -> Result<Column> {
        let raw = cudf_cxx::lists::ops::ffi::lists_index_of_scalar(&self.inner, &key.inner)
            .map_err(CudfError::from_cxx)?;
        Ok(Column { inner: raw })
    }

    /// Concatenate nested list elements within each row.
    ///
    /// Input must be a column of type LIST<LIST<T>>. The inner lists
    /// within each row are concatenated into a single list.
    pub fn lists_concatenate_list_elements(&self) -> Result<Column> {
        let raw = cudf_cxx::lists::ops::ffi::lists_concatenate_list_elements(&self.inner)
            .map_err(CudfError::from_cxx)?;
        Ok(Column { inner: raw })
    }

    /// Generate list column of arithmetic sequences.
    ///
    /// For each row, creates a sequence starting at `starts[i]`
    /// with `sizes[i]` elements incrementing by 1.
    pub fn lists_sequences(starts: &Column, sizes: &Column) -> Result<Column> {
        let raw = cudf_cxx::lists::ops::ffi::lists_sequences(&starts.inner, &sizes.inner)
            .map_err(CudfError::from_cxx)?;
        Ok(Column { inner: raw })
    }

    /// Gather elements from lists based on per-row gather maps.
    ///
    /// `gather_map` must be a lists column of INT32 indices.
    pub fn lists_segmented_gather(&self, gather_map: &Column) -> Result<Column> {
        let raw = cudf_cxx::lists::ops::ffi::lists_segmented_gather(&self.inner, &gather_map.inner)
            .map_err(CudfError::from_cxx)?;
        Ok(Column { inner: raw })
    }

    /// Check if lists at each row overlap with the corresponding row in `rhs`.
    ///
    /// Returns a BOOL8 column.
    pub fn lists_have_overlap(&self, rhs: &Column) -> Result<Column> {
        let raw = cudf_cxx::lists::ops::ffi::lists_have_overlap(&self.inner, &rhs.inner)
            .map_err(CudfError::from_cxx)?;
        Ok(Column { inner: raw })
    }

    /// Distinct elements common to both lists at each row.
    pub fn lists_intersect_distinct(&self, rhs: &Column) -> Result<Column> {
        let raw = cudf_cxx::lists::ops::ffi::lists_intersect_distinct(&self.inner, &rhs.inner)
            .map_err(CudfError::from_cxx)?;
        Ok(Column { inner: raw })
    }

    /// Distinct elements found in either list at each row.
    pub fn lists_union_distinct(&self, rhs: &Column) -> Result<Column> {
        let raw = cudf_cxx::lists::ops::ffi::lists_union_distinct(&self.inner, &rhs.inner)
            .map_err(CudfError::from_cxx)?;
        Ok(Column { inner: raw })
    }

    /// Distinct elements in this list but not in `rhs` at each row.
    pub fn lists_difference_distinct(&self, rhs: &Column) -> Result<Column> {
        let raw = cudf_cxx::lists::ops::ffi::lists_difference_distinct(&self.inner, &rhs.inner)
            .map_err(CudfError::from_cxx)?;
        Ok(Column { inner: raw })
    }

    /// Reverse elements within each list row.
    pub fn lists_reverse(&self) -> Result<Column> {
        let raw =
            cudf_cxx::lists::ops::ffi::lists_reverse(&self.inner).map_err(CudfError::from_cxx)?;
        Ok(Column { inner: raw })
    }

    /// Filter list elements using a boolean mask list column.
    ///
    /// `mask` must be a lists column of BOOL8 with the same structure.
    pub fn lists_apply_boolean_mask(&self, mask: &Column) -> Result<Column> {
        let raw = cudf_cxx::lists::ops::ffi::lists_apply_boolean_mask(&self.inner, &mask.inner)
            .map_err(CudfError::from_cxx)?;
        Ok(Column { inner: raw })
    }

    /// Remove duplicate elements within each list row.
    pub fn lists_distinct(&self) -> Result<Column> {
        let raw =
            cudf_cxx::lists::ops::ffi::lists_distinct(&self.inner).map_err(CudfError::from_cxx)?;
        Ok(Column { inner: raw })
    }

    /// Stable sort elements within each list row.
    ///
    /// Like [`lists_sort`](Self::lists_sort) but preserves relative order of equal elements.
    pub fn lists_stable_sort(&self, ascending: bool, null_order: NullOrder) -> Result<Column> {
        let raw =
            cudf_cxx::lists::ops::ffi::lists_stable_sort(&self.inner, ascending, null_order as i32)
                .map_err(CudfError::from_cxx)?;
        Ok(Column { inner: raw })
    }

    /// Extract elements using per-row indices from a column.
    ///
    /// For each row, extracts the element at the index specified by the
    /// corresponding value in `indices`. Null indices produce null output.
    pub fn lists_extract_column_index(&self, indices: &Column) -> Result<Column> {
        let raw =
            cudf_cxx::lists::ops::ffi::lists_extract_column_index(&self.inner, &indices.inner)
                .map_err(CudfError::from_cxx)?;
        Ok(Column { inner: raw })
    }

    /// Check whether each list row contains the corresponding value from a column.
    ///
    /// Returns a boolean column where `true` indicates the list row
    /// contains the value from the corresponding row in `search_keys`.
    pub fn lists_contains_column(&self, search_keys: &Column) -> Result<Column> {
        let raw = cudf_cxx::lists::ops::ffi::lists_contains_column(&self.inner, &search_keys.inner)
            .map_err(CudfError::from_cxx)?;
        Ok(Column { inner: raw })
    }
}

impl Table {
    /// Explode a list column with position indices.
    ///
    /// Like [`lists_explode`](Self::lists_explode) but includes an additional
    /// column with the position of each element within its original list.
    pub fn lists_explode_position(&self, explode_col_idx: usize) -> Result<Table> {
        let raw =
            cudf_cxx::lists::ops::ffi::lists_explode_position(&self.inner, explode_col_idx as i32)
                .map_err(CudfError::from_cxx)?;
        Ok(Table { inner: raw })
    }

    /// Explode outer with position indices.
    ///
    /// Like [`lists_explode_outer`](Self::lists_explode_outer) but includes
    /// position indices.
    pub fn lists_explode_outer_position(&self, explode_col_idx: usize) -> Result<Table> {
        let raw = cudf_cxx::lists::ops::ffi::lists_explode_outer_position(
            &self.inner,
            explode_col_idx as i32,
        )
        .map_err(CudfError::from_cxx)?;
        Ok(Table { inner: raw })
    }

    /// Concatenate lists across columns (row-wise).
    ///
    /// All columns must be list columns with the same child type.
    /// Returns a single list column where each row is the concatenation
    /// of lists from all input columns.
    pub fn lists_concatenate_rows(&self) -> Result<Column> {
        let raw = cudf_cxx::lists::ops::ffi::lists_concatenate_rows(&self.inner)
            .map_err(CudfError::from_cxx)?;
        Ok(Column { inner: raw })
    }
}
