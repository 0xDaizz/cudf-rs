//! GPU-accelerated rolling window operations.
//!
//! Provides fixed-size rolling window aggregation for [`Column`]s.
//!
//! # Examples
//!
//! ```rust,no_run
//! use cudf::Column;
//! use cudf::rolling::RollingAgg;
//!
//! let col = Column::from_slice(&[1.0f64, 2.0, 3.0, 4.0, 5.0]).unwrap();
//! let rolling_mean = col.rolling(3, 0, 1, RollingAgg::Mean).unwrap();
//! ```

use crate::column::Column;
use crate::error::{CudfError, Result};
use crate::table::Table;
use crate::types::checked_i32;

/// Aggregation type for rolling window operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RollingAgg {
    /// Rolling sum.
    Sum = 0,
    /// Rolling minimum.
    Min = 1,
    /// Rolling maximum.
    Max = 2,
    /// Rolling count of non-null values.
    Count = 3,
    /// Rolling mean.
    Mean = 4,
    /// Collect values into a list.
    CollectList = 5,
    /// Row number within the window.
    RowNumber = 6,
    /// Lead (next value).
    Lead = 7,
    /// Lag (previous value).
    Lag = 8,
}

impl Column {
    /// Apply a fixed-size rolling window aggregation.
    ///
    /// # Arguments
    ///
    /// * `preceding` - Number of rows before the current row to include
    ///   in the window (including the current row for some aggregations).
    /// * `following` - Number of rows after the current row to include.
    /// * `min_periods` - Minimum number of non-null observations required
    ///   to produce a non-null result.
    /// * `agg` - The aggregation to apply over each window.
    ///
    /// # Errors
    ///
    /// Returns an error if the operation is unsupported for the column type,
    /// if window parameters are invalid, or if a GPU error occurs.
    pub fn rolling(
        &self,
        preceding: usize,
        following: usize,
        min_periods: usize,
        agg: RollingAgg,
    ) -> Result<Column> {
        let raw = cudf_cxx::rolling::ffi::rolling_window(
            &self.inner,
            checked_i32(preceding)?,
            checked_i32(following)?,
            checked_i32(min_periods)?,
            agg as i32,
        )
        .map_err(CudfError::from_cxx)?;

        Ok(Column { inner: raw })
    }

    /// Apply a grouped rolling window aggregation.
    ///
    /// The input column and group keys must be pre-sorted by the group keys.
    /// Within each group (defined by `group_keys`), the rolling window is
    /// applied independently.
    ///
    /// # Arguments
    ///
    /// * `group_keys` - Table of key columns that define groups (must be sorted).
    /// * `preceding` - Number of rows before the current row in the window.
    /// * `following` - Number of rows after the current row in the window.
    /// * `min_periods` - Minimum non-null observations required.
    /// * `agg` - The aggregation to apply over each window.
    ///
    /// # Errors
    ///
    /// Returns an error if the operation is unsupported, parameters are invalid,
    /// or a GPU error occurs.
    pub fn grouped_rolling(
        &self,
        group_keys: &Table,
        preceding: usize,
        following: usize,
        min_periods: usize,
        agg: RollingAgg,
    ) -> Result<Column> {
        let raw = cudf_cxx::rolling::ffi::grouped_rolling_window(
            &group_keys.inner,
            &self.inner,
            checked_i32(preceding)?,
            checked_i32(following)?,
            checked_i32(min_periods)?,
            agg as i32,
        )
        .map_err(CudfError::from_cxx)?;

        Ok(Column { inner: raw })
    }

    /// Apply a variable-size rolling window aggregation.
    ///
    /// Unlike [`rolling`](Self::rolling), the window size can vary per row.
    /// `preceding_col` and `following_col` are integer columns specifying
    /// the window sizes for each element.
    ///
    /// # Errors
    ///
    /// Returns an error if the operation is unsupported, window columns
    /// have mismatched sizes, or a GPU error occurs.
    pub fn rolling_variable(
        &self,
        preceding_col: &Column,
        following_col: &Column,
        min_periods: usize,
        agg: RollingAgg,
    ) -> Result<Column> {
        let raw = cudf_cxx::rolling::ffi::rolling_window_variable(
            &self.inner,
            &preceding_col.inner,
            &following_col.inner,
            checked_i32(min_periods)?,
            agg as i32,
        )
        .map_err(CudfError::from_cxx)?;

        Ok(Column { inner: raw })
    }
}

/// Check if a rolling aggregation is valid for a given source data type and rolling aggregation.
///
/// Returns `true` if the aggregation can be applied.
pub fn is_valid_rolling_aggregation(
    source_type: crate::types::DataType,
    agg: RollingAgg,
) -> crate::error::Result<bool> {
    cudf_cxx::rolling::ffi::is_valid_rolling_aggregation(source_type.id() as i32, agg as i32)
        .map_err(crate::error::CudfError::from_cxx)
}
