//! GPU-accelerated groupby aggregation.
//!
//! Provides a builder-pattern API for performing groupby operations on
//! GPU-resident tables.
//!
//! # Examples
//!
//! ```rust,no_run
//! use cudf::{Column, Table};
//! use cudf::groupby::GroupBy;
//! use cudf::aggregation::AggregationKind;
//!
//! // Keys: group labels
//! let keys_col = Column::from_slice(&[1i32, 1, 2, 2, 3]).unwrap();
//! let keys = Table::new(vec![keys_col]).unwrap();
//!
//! // Values: columns to aggregate
//! let val_a = Column::from_slice(&[10.0f64, 20.0, 30.0, 40.0, 50.0]).unwrap();
//! let val_b = Column::from_slice(&[1i32, 2, 3, 4, 5]).unwrap();
//! let values = Table::new(vec![val_a, val_b]).unwrap();
//!
//! let result = GroupBy::new(&keys)
//!     .agg(0, AggregationKind::Sum)
//!     .agg(1, AggregationKind::Mean)
//!     .execute(&values)
//!     .unwrap();
//! ```

use crate::aggregation::{Aggregation, AggregationKind};
use crate::error::{CudfError, Result};
use crate::table::Table;

/// A groupby operation builder.
///
/// Accumulates aggregation requests and executes them against a values table.
/// The result is a [`Table`] with key columns first, followed by aggregation
/// result columns in the order they were added.
pub struct GroupBy<'a> {
    keys: &'a Table,
    requests: Vec<(usize, Aggregation)>,
}

impl<'a> GroupBy<'a> {
    /// Create a new groupby builder with the given key columns.
    pub fn new(keys: &'a Table) -> Self {
        Self {
            keys,
            requests: Vec::new(),
        }
    }

    /// Add an aggregation request for a value column.
    ///
    /// `column` is the 0-based index into the values table.
    /// `kind` specifies the aggregation to compute.
    ///
    /// Returns `self` for method chaining.
    pub fn agg(mut self, column: usize, kind: AggregationKind) -> Self {
        self.requests.push((column, Aggregation::new(kind)));
        self
    }

    /// Add a pre-built [`Aggregation`] for a value column.
    ///
    /// Use this when you need to configure null handling or other options
    /// before creating the aggregation.
    pub fn agg_with(mut self, column: usize, aggregation: Aggregation) -> Self {
        self.requests.push((column, aggregation));
        self
    }

    /// Execute the groupby, returning a table with key columns followed by
    /// aggregation result columns.
    ///
    /// # Errors
    ///
    /// Returns an error if column indices are out of bounds, aggregation
    /// types are incompatible with column types, or a GPU error occurs.
    pub fn execute(self, values: &Table) -> Result<Table> {
        if self.requests.is_empty() {
            return Err(CudfError::InvalidArgument(
                "groupby requires at least one aggregation request".to_string(),
            ));
        }

        let mut builder =
            cudf_cxx::groupby::ffi::groupby_new(&self.keys.inner);

        for (col_idx, agg) in self.requests {
            cudf_cxx::groupby::ffi::groupby_add_request(
                builder.pin_mut(),
                col_idx as i32,
                agg.inner,
            );
        }

        let raw = cudf_cxx::groupby::ffi::groupby_execute(
            builder.pin_mut(),
            &values.inner,
        )
        .map_err(CudfError::from_cxx)?;

        Ok(Table { inner: raw })
    }

    /// Execute the groupby and return only the key columns result.
    ///
    /// Useful when you need just the unique group keys.
    pub fn execute_keys(self, values: &Table) -> Result<Table> {
        if self.requests.is_empty() {
            return Err(CudfError::InvalidArgument(
                "groupby requires at least one aggregation request".to_string(),
            ));
        }

        let mut builder =
            cudf_cxx::groupby::ffi::groupby_new(&self.keys.inner);

        for (col_idx, agg) in self.requests {
            cudf_cxx::groupby::ffi::groupby_add_request(
                builder.pin_mut(),
                col_idx as i32,
                agg.inner,
            );
        }

        let raw = cudf_cxx::groupby::ffi::groupby_execute_keys(
            builder.pin_mut(),
            &values.inner,
        )
        .map_err(CudfError::from_cxx)?;

        Ok(Table { inner: raw })
    }

    /// Execute the groupby and return only the aggregation result columns.
    pub fn execute_values(self, values: &Table) -> Result<Table> {
        if self.requests.is_empty() {
            return Err(CudfError::InvalidArgument(
                "groupby requires at least one aggregation request".to_string(),
            ));
        }

        let mut builder =
            cudf_cxx::groupby::ffi::groupby_new(&self.keys.inner);

        for (col_idx, agg) in self.requests {
            cudf_cxx::groupby::ffi::groupby_add_request(
                builder.pin_mut(),
                col_idx as i32,
                agg.inner,
            );
        }

        let raw = cudf_cxx::groupby::ffi::groupby_execute_values(
            builder.pin_mut(),
            &values.inner,
        )
        .map_err(CudfError::from_cxx)?;

        Ok(Table { inner: raw })
    }
}
