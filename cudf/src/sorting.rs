//! GPU-accelerated sorting operations.
//!
//! Provides sorting, ranking, and order-checking for [`Table`]s and [`Column`]s.
//!
//! # Examples
//!
//! ```rust,no_run
//! use cudf::{Column, Table};
//! use cudf::sorting::{SortOrder, NullOrder};
//!
//! let col = Column::from_slice(&[3i32, 1, 2]).unwrap();
//! let table = Table::new(vec![col]).unwrap();
//!
//! let sorted = table.sort(
//!     &[SortOrder::Ascending],
//!     &[NullOrder::After],
//! ).unwrap();
//! ```

use crate::column::Column;
use crate::error::{CudfError, Result};
use crate::table::Table;
pub use crate::types::NullHandling;

/// Sort direction for a column.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SortOrder {
    /// Sort in ascending order.
    Ascending = 0,
    /// Sort in descending order.
    Descending = 1,
}

/// Where null values appear in sorted output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NullOrder {
    /// Nulls appear after all non-null values.
    After = 0,
    /// Nulls appear before all non-null values.
    Before = 1,
}

/// Method used to compute ranks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RankMethod {
    /// Ranks are assigned in order of appearance.
    First = 0,
    /// Average of the ranks for tied values.
    Average = 1,
    /// Minimum rank for tied values.
    Min = 2,
    /// Maximum rank for tied values.
    Max = 3,
    /// Dense ranking (no gaps between ranks).
    Dense = 4,
}

// -- Helper conversions --

fn sort_orders_to_i32(orders: &[SortOrder]) -> Vec<i32> {
    orders.iter().map(|o| *o as i32).collect()
}

fn null_orders_to_i32(orders: &[NullOrder]) -> Vec<i32> {
    orders.iter().map(|o| *o as i32).collect()
}

// -- Table methods --

impl Table {
    /// Sort this table by its columns, returning a new sorted table.
    ///
    /// `column_order` and `null_order` must have one entry per column.
    ///
    /// # Errors
    ///
    /// Returns an error if the slice lengths don't match the number of columns,
    /// or if a GPU error occurs.
    pub fn sort(&self, column_order: &[SortOrder], null_order: &[NullOrder]) -> Result<Table> {
        self.validate_order_slices(column_order.len(), null_order.len())?;

        let co = sort_orders_to_i32(column_order);
        let no = null_orders_to_i32(null_order);

        let raw =
            cudf_cxx::sorting::ffi::sort(&self.inner, &co, &no).map_err(CudfError::from_cxx)?;

        Ok(Table { inner: raw })
    }

    /// Returns a column of row indices that would sort this table.
    ///
    /// The returned column contains `i32` indices suitable for use with
    /// [`Table::gather`](crate::table::Table::gather).
    pub fn sorted_order(
        &self,
        column_order: &[SortOrder],
        null_order: &[NullOrder],
    ) -> Result<Column> {
        self.validate_order_slices(column_order.len(), null_order.len())?;

        let co = sort_orders_to_i32(column_order);
        let no = null_orders_to_i32(null_order);

        let raw = cudf_cxx::sorting::ffi::sorted_order(&self.inner, &co, &no)
            .map_err(CudfError::from_cxx)?;

        Ok(Column { inner: raw })
    }

    /// Check whether this table is sorted according to the given order.
    pub fn is_sorted(&self, column_order: &[SortOrder], null_order: &[NullOrder]) -> Result<bool> {
        self.validate_order_slices(column_order.len(), null_order.len())?;

        let co = sort_orders_to_i32(column_order);
        let no = null_orders_to_i32(null_order);

        cudf_cxx::sorting::ffi::is_sorted(&self.inner, &co, &no).map_err(CudfError::from_cxx)
    }

    /// Validate that order slices match the number of columns.
    fn validate_order_slices(&self, co_len: usize, no_len: usize) -> Result<()> {
        let ncols = self.num_columns();
        if co_len != ncols {
            return Err(CudfError::InvalidArgument(format!(
                "column_order length ({}) must match num_columns ({})",
                co_len, ncols
            )));
        }
        if no_len != ncols {
            return Err(CudfError::InvalidArgument(format!(
                "null_order length ({}) must match num_columns ({})",
                no_len, ncols
            )));
        }
        Ok(())
    }
}

// -- Column methods --

impl Column {
    /// Compute the rank of each element in this column.
    ///
    /// Returns a column of `f64` rank values.
    pub fn rank(
        &self,
        method: RankMethod,
        order: SortOrder,
        null_order: NullOrder,
    ) -> Result<Column> {
        let raw = cudf_cxx::sorting::ffi::rank(
            &self.inner,
            method as i32,
            order as i32,
            null_order as i32,
            NullHandling::Include as i32,
            false,
        )
        .map_err(CudfError::from_cxx)?;

        Ok(Column { inner: raw })
    }
}
