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
use crate::column::Column;
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

        let mut builder = cudf_cxx::groupby::ffi::groupby_new(&self.keys.inner);

        for (col_idx, agg) in self.requests {
            cudf_cxx::groupby::ffi::groupby_add_request(
                builder.pin_mut(),
                col_idx as i32,
                agg.inner,
            );
        }

        let raw = cudf_cxx::groupby::ffi::groupby_execute(builder.pin_mut(), &values.inner)
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

        let mut builder = cudf_cxx::groupby::ffi::groupby_new(&self.keys.inner);

        for (col_idx, agg) in self.requests {
            cudf_cxx::groupby::ffi::groupby_add_request(
                builder.pin_mut(),
                col_idx as i32,
                agg.inner,
            );
        }

        let raw = cudf_cxx::groupby::ffi::groupby_execute_keys(builder.pin_mut(), &values.inner)
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

        let mut builder = cudf_cxx::groupby::ffi::groupby_new(&self.keys.inner);

        for (col_idx, agg) in self.requests {
            cudf_cxx::groupby::ffi::groupby_add_request(
                builder.pin_mut(),
                col_idx as i32,
                agg.inner,
            );
        }

        let raw = cudf_cxx::groupby::ffi::groupby_execute_values(builder.pin_mut(), &values.inner)
            .map_err(CudfError::from_cxx)?;

        Ok(Table { inner: raw })
    }
}

// ── GroupBy Scan ──────────────────────────────────────────────

/// Aggregation type for groupby scan operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GroupByScanOp {
    /// Cumulative sum within group.
    Sum = 0,
    /// Cumulative min within group.
    Min = 2,
    /// Cumulative max within group.
    Max = 3,
    /// Cumulative count within group.
    Count = 11,
    /// Rank within group.
    Rank = 12,
}

/// A builder for groupby scan operations.
///
/// Groupby scan computes cumulative (prefix) aggregations within each group.
///
/// # Examples
///
/// ```rust,no_run
/// use cudf::{Column, Table};
/// use cudf::groupby::{GroupByScan, GroupByScanOp};
///
/// let keys = Table::new(vec![
///     Column::from_slice(&[1i32, 1, 2, 2]).unwrap(),
/// ]).unwrap();
/// let values = Table::new(vec![
///     Column::from_slice(&[10.0f64, 20.0, 30.0, 40.0]).unwrap(),
/// ]).unwrap();
///
/// let result = GroupByScan::new(&keys)
///     .scan(0, GroupByScanOp::Sum)
///     .execute(&values)
///     .unwrap();
/// ```
pub struct GroupByScan<'a> {
    keys: &'a Table,
    requests: Vec<(usize, GroupByScanOp)>,
}

impl<'a> GroupByScan<'a> {
    /// Create a new groupby scan builder with the given key columns.
    pub fn new(keys: &'a Table) -> Self {
        Self {
            keys,
            requests: Vec::new(),
        }
    }

    /// Add a scan request for a value column.
    pub fn scan(mut self, column: usize, op: GroupByScanOp) -> Self {
        self.requests.push((column, op));
        self
    }

    /// Execute the grouped scan.
    ///
    /// Returns a table with key columns followed by scan result columns.
    pub fn execute(self, values: &Table) -> Result<Table> {
        if self.requests.is_empty() {
            return Err(CudfError::InvalidArgument(
                "groupby scan requires at least one request".to_string(),
            ));
        }

        let mut builder = cudf_cxx::groupby::ffi::groupby_scan_new(&self.keys.inner);

        for (col_idx, op) in self.requests {
            cudf_cxx::groupby::ffi::groupby_scan_add_request(
                builder.pin_mut(),
                col_idx as i32,
                op as i32,
            );
        }

        let raw = cudf_cxx::groupby::ffi::groupby_scan_execute(builder.pin_mut(), &values.inner)
            .map_err(CudfError::from_cxx)?;

        Ok(Table { inner: raw })
    }
}

// ── GroupBy Get Groups ────────────────────────────────────────

/// Result of a groupby get_groups operation.
pub struct GroupByGroups {
    /// The grouped key rows.
    pub keys: Table,
    /// Group boundary offsets (length = num_groups + 1).
    pub offsets: Column,
    /// The grouped values (if values were provided).
    pub values: Option<Table>,
}

/// Policy for replacing nulls within groups.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GroupByReplacePolicy {
    /// Fill with the preceding non-null value.
    Forward = 0,
    /// Fill with the following non-null value.
    Backward = 1,
}

impl Table {
    /// Get the group structure for this table used as keys.
    ///
    /// Returns the grouped keys and group boundary offsets.
    pub fn groupby_get_groups(&self) -> Result<GroupByGroups> {
        let mut raw =
            cudf_cxx::groupby::ffi::groupby_get_groups(&self.inner).map_err(CudfError::from_cxx)?;
        let keys_raw = cudf_cxx::groupby::ffi::groupby_groups_take_keys(raw.pin_mut())
            .map_err(CudfError::from_cxx)?;
        let offsets_raw = cudf_cxx::groupby::ffi::groupby_groups_take_offsets(raw.pin_mut())
            .map_err(CudfError::from_cxx)?;
        Ok(GroupByGroups {
            keys: Table { inner: keys_raw },
            offsets: Column { inner: offsets_raw },
            values: None,
        })
    }

    /// Get the group structure, also reordering `values` by group.
    ///
    /// Returns grouped keys, offsets, and the values table reordered
    /// so rows in the same group are contiguous.
    pub fn groupby_get_groups_with_values(&self, values: &Table) -> Result<GroupByGroups> {
        let mut raw =
            cudf_cxx::groupby::ffi::groupby_get_groups_with_values(&self.inner, &values.inner)
                .map_err(CudfError::from_cxx)?;
        let keys_raw = cudf_cxx::groupby::ffi::groupby_groups_take_keys(raw.pin_mut())
            .map_err(CudfError::from_cxx)?;
        let offsets_raw = cudf_cxx::groupby::ffi::groupby_groups_take_offsets(raw.pin_mut())
            .map_err(CudfError::from_cxx)?;
        let values_raw = cudf_cxx::groupby::ffi::groupby_groups_take_values(raw.pin_mut())
            .map_err(CudfError::from_cxx)?;
        Ok(GroupByGroups {
            keys: Table { inner: keys_raw },
            offsets: Column { inner: offsets_raw },
            values: Some(Table { inner: values_raw }),
        })
    }

    /// Replace nulls within groups using forward or backward fill.
    ///
    /// `policies` must have one entry per column in `values`.
    ///
    /// Returns a table with key columns followed by null-replaced value columns.
    pub fn groupby_replace_nulls(
        &self,
        values: &Table,
        policies: &[GroupByReplacePolicy],
    ) -> Result<Table> {
        let p: Vec<i32> = policies.iter().map(|&p| p as i32).collect();
        let raw = cudf_cxx::groupby::ffi::groupby_replace_nulls(&self.inner, &values.inner, &p)
            .map_err(CudfError::from_cxx)?;
        Ok(Table { inner: raw })
    }
}
