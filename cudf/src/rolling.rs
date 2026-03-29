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
            preceding as i32,
            following as i32,
            min_periods as i32,
            agg as i32,
        )
        .map_err(CudfError::from_cxx)?;

        Ok(Column { inner: raw })
    }
}
