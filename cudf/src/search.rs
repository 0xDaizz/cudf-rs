//! GPU-accelerated search operations.
//!
//! Provides binary search and containment checks for [`Table`]s and [`Column`]s.
//!
//! # Examples
//!
//! ```rust,no_run
//! use cudf::{Column, Table};
//! use cudf::sorting::{SortOrder, NullOrder};
//!
//! let sorted = Table::new(vec![Column::from_slice(&[1i32, 2, 3, 4, 5]).unwrap()]).unwrap();
//! let values = Table::new(vec![Column::from_slice(&[2i32, 4]).unwrap()]).unwrap();
//!
//! let lb = sorted.lower_bound(
//!     &values,
//!     &[SortOrder::Ascending],
//!     &[NullOrder::After],
//! ).unwrap();
//! ```

use crate::column::Column;
use crate::error::{CudfError, Result};
use crate::sorting::{NullOrder, SortOrder};
use crate::table::Table;

impl Table {
    /// Find the lower bound indices for each row in `values` within this sorted table.
    ///
    /// Returns a column of `i32` indices. This table must be pre-sorted
    /// according to the specified order.
    pub fn lower_bound(
        &self,
        values: &Table,
        orders: &[SortOrder],
        null_orders: &[NullOrder],
    ) -> Result<Column> {
        self.validate_order_slices(orders.len(), null_orders.len())?;
        let ord: Vec<i32> = orders.iter().map(|o| *o as i32).collect();
        let nul: Vec<i32> = null_orders.iter().map(|o| *o as i32).collect();

        let raw = cudf_cxx::search::ffi::lower_bound(&self.inner, &values.inner, &ord, &nul)
            .map_err(CudfError::from_cxx)?;
        Ok(Column { inner: raw })
    }

    /// Find the upper bound indices for each row in `values` within this sorted table.
    ///
    /// Returns a column of `i32` indices. This table must be pre-sorted
    /// according to the specified order.
    pub fn upper_bound(
        &self,
        values: &Table,
        orders: &[SortOrder],
        null_orders: &[NullOrder],
    ) -> Result<Column> {
        self.validate_order_slices(orders.len(), null_orders.len())?;
        let ord: Vec<i32> = orders.iter().map(|o| *o as i32).collect();
        let nul: Vec<i32> = null_orders.iter().map(|o| *o as i32).collect();

        let raw = cudf_cxx::search::ffi::upper_bound(&self.inner, &values.inner, &ord, &nul)
            .map_err(CudfError::from_cxx)?;
        Ok(Column { inner: raw })
    }
}

impl Column {
    /// For each element in `needles`, check if it exists in this column.
    ///
    /// Returns a boolean column where `true` indicates the needle was found
    /// in this column (the haystack).
    pub fn contains(&self, needles: &Column) -> Result<Column> {
        let raw = cudf_cxx::search::ffi::contains_column(&self.inner, &needles.inner)
            .map_err(CudfError::from_cxx)?;
        Ok(Column { inner: raw })
    }

    /// Check if a scalar value exists in this column.
    ///
    /// Returns `true` if the scalar is found in the column.
    pub fn contains_scalar(&self, needle: &crate::scalar::Scalar) -> Result<bool> {
        cudf_cxx::search::ffi::contains_scalar(&self.inner, &needle.inner)
            .map_err(CudfError::from_cxx)
    }
}
