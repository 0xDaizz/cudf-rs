//! GPU-accelerated quantile operations.
//!
//! Provides quantile computation for [`Column`]s and [`Table`]s.
//!
//! # Examples
//!
//! ```rust,no_run
//! use cudf::Column;
//! use cudf::quantiles::Interpolation;
//!
//! let col = Column::from_slice(&[1.0f64, 2.0, 3.0, 4.0, 5.0]).unwrap();
//! let q = col.quantile(&[0.25, 0.5, 0.75], Interpolation::Linear).unwrap();
//! ```

use crate::column::Column;
use crate::error::{CudfError, Result};
use crate::sorting::{NullOrder, SortOrder};
use crate::table::Table;

/// Interpolation method for quantile computation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Interpolation {
    /// Linear interpolation between data points.
    Linear = 0,
    /// Use the lower data point.
    Lower = 1,
    /// Use the higher data point.
    Higher = 2,
    /// Use the midpoint of the two data points.
    Midpoint = 3,
    /// Use the nearest data point.
    Nearest = 4,
}

impl Column {
    /// Compute quantile(s) of this column.
    ///
    /// Returns a column containing one value per requested quantile.
    ///
    /// # Arguments
    ///
    /// * `q` - Slice of quantile values in `[0, 1]`.
    /// * `interp` - Interpolation method to use.
    ///
    /// # Errors
    ///
    /// Returns an error if the column type doesn't support quantiles
    /// or if a GPU error occurs.
    pub fn quantile(&self, q: &[f64], interp: Interpolation) -> Result<Column> {
        let raw = cudf_cxx::quantiles::ffi::quantile(
            &self.inner,
            q,
            interp as i32,
        )
        .map_err(CudfError::from_cxx)?;

        Ok(Column { inner: raw })
    }

    /// Compute percentile approximation using t-digest.
    ///
    /// This column must contain t-digest data (typically produced by
    /// a t-digest aggregation).
    ///
    /// # Arguments
    ///
    /// * `percentiles` - Slice of percentile values in `[0, 1]`.
    ///
    /// # Errors
    ///
    /// Returns an error if the column is not a valid t-digest column
    /// or if a GPU error occurs.
    pub fn percentile_approx(&self, percentiles: &[f64]) -> Result<Column> {
        let raw = cudf_cxx::quantiles::ffi::percentile_approx(
            &self.inner,
            percentiles,
        )
        .map_err(CudfError::from_cxx)?;

        Ok(Column { inner: raw })
    }
}

impl Table {
    /// Compute quantiles of this table (row-wise).
    ///
    /// Returns a table containing the rows at the requested quantile positions.
    ///
    /// # Arguments
    ///
    /// * `q` - Slice of quantile values in `[0, 1]`.
    /// * `interp` - Interpolation method to use.
    /// * `is_sorted` - Whether the input is already sorted.
    /// * `column_order` - Sort order per column (used if input is sorted).
    /// * `null_order` - Null ordering per column (used if input is sorted).
    ///
    /// # Errors
    ///
    /// Returns an error if the arguments are invalid or if a GPU error occurs.
    pub fn quantiles(
        &self,
        q: &[f64],
        interp: Interpolation,
        is_sorted: bool,
        column_order: &[SortOrder],
        null_order: &[NullOrder],
    ) -> Result<Table> {
        let orders: Vec<i32> = column_order.iter().map(|o| *o as i32).collect();
        let null_orders: Vec<i32> = null_order.iter().map(|o| *o as i32).collect();

        let raw = cudf_cxx::quantiles::ffi::quantiles_table(
            &self.inner,
            q,
            interp as i32,
            is_sorted,
            &orders,
            &null_orders,
        )
        .map_err(CudfError::from_cxx)?;

        Ok(Table { inner: raw })
    }
}
