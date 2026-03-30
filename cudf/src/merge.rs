//! GPU-accelerated merge operations.
//!
//! Merges two pre-sorted [`Table`]s into a single sorted table.
//!
//! # Examples
//!
//! ```rust,no_run
//! use cudf::{Column, Table};
//! use cudf::sorting::{SortOrder, NullOrder};
//!
//! let left = Table::new(vec![Column::from_slice(&[1i32, 3, 5]).unwrap()]).unwrap();
//! let right = Table::new(vec![Column::from_slice(&[2i32, 4, 6]).unwrap()]).unwrap();
//!
//! let merged = left.merge(
//!     &right,
//!     &[0],
//!     &[SortOrder::Ascending],
//!     &[NullOrder::After],
//! ).unwrap();
//! assert_eq!(merged.num_rows(), 6);
//! ```

use crate::error::{CudfError, Result};
use crate::sorting::{NullOrder, SortOrder};
use crate::table::Table;
use crate::types::checked_i32;

impl Table {
    /// Merge this pre-sorted table with another pre-sorted table.
    ///
    /// Both tables must be sorted by the specified key columns in the
    /// given order. The result is a single sorted table.
    ///
    /// # Arguments
    ///
    /// * `other` - The other pre-sorted table to merge with.
    /// * `key_cols` - Column indices to merge on.
    /// * `orders` - Sort direction for each key column.
    /// * `null_orders` - Null placement for each key column.
    pub fn merge(
        &self,
        other: &Table,
        key_cols: &[usize],
        orders: &[SortOrder],
        null_orders: &[NullOrder],
    ) -> Result<Table> {
        if key_cols.is_empty() {
            return Err(CudfError::InvalidArgument(
                "merge requires at least one key column".to_string(),
            ));
        }
        if orders.len() != key_cols.len() {
            return Err(CudfError::InvalidArgument(format!(
                "column_order length ({}) must match key_cols length ({})",
                orders.len(),
                key_cols.len()
            )));
        }
        if null_orders.len() != key_cols.len() {
            return Err(CudfError::InvalidArgument(format!(
                "null_order length ({}) must match key_cols length ({})",
                null_orders.len(),
                key_cols.len()
            )));
        }
        let keys: Vec<i32> = key_cols.iter().map(|&k| checked_i32(k)).collect::<Result<Vec<i32>>>()?;
        let ord: Vec<i32> = orders.iter().map(|o| *o as i32).collect();
        let nul: Vec<i32> = null_orders.iter().map(|o| *o as i32).collect();

        let raw = cudf_cxx::merge::ffi::merge_tables(&self.inner, &other.inner, &keys, &ord, &nul)
            .map_err(CudfError::from_cxx)?;
        Ok(Table { inner: raw })
    }
}
