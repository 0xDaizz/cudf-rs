//! GPU-accelerated reduction operations.
//!
//! Provides reduce, scan (prefix-sum), and segmented reduce for [`Column`]s.
//!
//! # Examples
//!
//! ```rust,no_run
//! use cudf::Column;
//! use cudf::reduction::ReduceOp;
//! use cudf::types::DataType;
//! use cudf::TypeId;
//!
//! let col = Column::from_slice(&[1i32, 2, 3, 4]).unwrap();
//! let sum = col.reduce(ReduceOp::Sum, DataType::new(TypeId::Int64)).unwrap();
//! assert!(sum.is_valid());
//! ```

use crate::column::Column;
use crate::error::{CudfError, Result};
use crate::scalar::Scalar;
use crate::types::DataType;

/// Aggregation operations for column reduction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReduceOp {
    /// Sum of all values.
    Sum = 0,
    /// Product of all values.
    Product = 1,
    /// Minimum value.
    Min = 2,
    /// Maximum value.
    Max = 3,
    /// Sum of squares of all values.
    SumOfSquares = 4,
    /// Arithmetic mean.
    Mean = 5,
    /// Variance.
    Variance = 6,
    /// Standard deviation.
    Std = 7,
    /// Logical OR (true if any value is true).
    Any = 8,
    /// Logical AND (true if all values are true).
    All = 9,
    /// Median value.
    Median = 10,
}

/// Aggregation operations for prefix scan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScanOp {
    /// Cumulative sum.
    Sum = 0,
    /// Cumulative product.
    Product = 1,
    /// Cumulative minimum.
    Min = 2,
    /// Cumulative maximum.
    Max = 3,
}

/// Result of a [`Column::minmax`] operation.
pub struct MinMaxResult {
    /// The minimum value.
    pub min: Scalar,
    /// The maximum value.
    pub max: Scalar,
}

impl Column {
    /// Compute the minimum and maximum values simultaneously.
    ///
    /// This is more efficient than calling `reduce(Min)` and `reduce(Max)`
    /// separately, as it requires only a single pass over the data.
    ///
    /// # Errors
    ///
    /// Returns an error if the column type does not support comparison
    /// or if a GPU error occurs.
    pub fn minmax(&self) -> Result<MinMaxResult> {
        let mut raw = cudf_cxx::reduction::ffi::minmax(&self.inner).map_err(CudfError::from_cxx)?;
        let min_raw = cudf_cxx::reduction::ffi::minmax_take_min(raw.pin_mut())
            .map_err(CudfError::from_cxx)?;
        let max_raw = cudf_cxx::reduction::ffi::minmax_take_max(raw.pin_mut())
            .map_err(CudfError::from_cxx)?;
        Ok(MinMaxResult {
            min: Scalar { inner: min_raw },
            max: Scalar { inner: max_raw },
        })
    }

    /// Reduce the entire column to a single scalar value.
    ///
    /// # Arguments
    ///
    /// * `op` - The reduction operation to apply.
    /// * `output_type` - The desired output scalar data type.
    ///
    /// # Errors
    ///
    /// Returns an error if the operation is unsupported for the column type
    /// or if a GPU error occurs.
    pub fn reduce(&self, op: ReduceOp, output_type: DataType) -> Result<Scalar> {
        let raw = cudf_cxx::reduction::ffi::reduce(&self.inner, op as i32, output_type.id() as i32)
            .map_err(CudfError::from_cxx)?;

        Ok(Scalar { inner: raw })
    }

    /// Compute a prefix scan (cumulative operation) over this column.
    ///
    /// # Arguments
    ///
    /// * `op` - The scan operation to apply.
    /// * `inclusive` - If true, element i includes itself in the result.
    ///   If false (exclusive), element i includes only elements before it.
    ///
    /// # Errors
    ///
    /// Returns an error if the operation is unsupported for the column type
    /// or if a GPU error occurs.
    pub fn scan(&self, op: ScanOp, inclusive: bool) -> Result<Column> {
        let raw = cudf_cxx::reduction::ffi::scan(&self.inner, op as i32, inclusive)
            .map_err(CudfError::from_cxx)?;

        Ok(Column { inner: raw })
    }

    /// Reduce within segments defined by an offsets column.
    ///
    /// The offsets column contains `n+1` values defining `n` segments:
    /// segment `i` covers elements `[offsets[i], offsets[i+1])`.
    ///
    /// # Arguments
    ///
    /// * `offsets` - Column of segment boundaries (int32).
    /// * `op` - The reduction operation to apply within each segment.
    /// * `output_type` - The desired output column data type.
    /// * `include_nulls` - Whether to include null values in the reduction.
    ///
    /// # Errors
    ///
    /// Returns an error if the arguments are invalid or if a GPU error occurs.
    pub fn segmented_reduce(
        &self,
        offsets: &Column,
        op: ReduceOp,
        output_type: DataType,
        include_nulls: bool,
    ) -> Result<Column> {
        let raw = cudf_cxx::reduction::ffi::segmented_reduce(
            &self.inner,
            &offsets.inner,
            op as i32,
            output_type.id() as i32,
            include_nulls,
        )
        .map_err(CudfError::from_cxx)?;

        Ok(Column { inner: raw })
    }
}
