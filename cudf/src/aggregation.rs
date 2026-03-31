//! Aggregation kinds for groupby and other reduction operations.
//!
//! [`AggregationKind`] is a Rust enum that maps to libcudf's aggregation factory
//! functions. Use it with [`GroupBy`](crate::groupby::GroupBy) to specify how
//! each value column should be aggregated.
//!
//! # Examples
//!
//! ```rust,no_run
//! use cudf::aggregation::AggregationKind;
//!
//! let sum = AggregationKind::Sum;
//! let var = AggregationKind::Variance { ddof: 1 };
//! let q50 = AggregationKind::Quantile { q: 0.5 };
//! ```

use cxx::UniquePtr;

use crate::types::NullHandling;

/// The kind of aggregation to perform.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub enum AggregationKind {
    /// Sum of non-null values.
    Sum,
    /// Product of non-null values.
    Product,
    /// Minimum value.
    Min,
    /// Maximum value.
    Max,
    /// Count of all values (including nulls).
    Count,
    /// Count of non-null values only.
    CountValid,
    /// Logical OR (any true).
    Any,
    /// Logical AND (all true).
    All,
    /// Sum of squares.
    SumOfSquares,
    /// Arithmetic mean.
    Mean,
    /// Median value.
    Median,
    /// Variance with the given degrees of freedom correction.
    Variance { ddof: i32 },
    /// Standard deviation with the given degrees of freedom correction.
    Std { ddof: i32 },
    /// Count of unique non-null values.
    Nunique,
    /// The nth element (0-indexed).
    NthElement { n: i32 },
    /// Collect values into a list.
    CollectList,
    /// Collect unique values into a set.
    CollectSet,
    /// Index of the maximum value.
    Argmax,
    /// Index of the minimum value.
    Argmin,
    /// Row number (1-based rank by order of appearance).
    ///
    /// **Note:** This is a scan-only aggregation. Using it with `GroupBy::aggregate`
    /// will return an error. Use [`GroupByScan`](crate::groupby::GroupByScan) instead.
    RowNumber,
    /// Quantile at the given probability.
    Quantile { q: f64 },
    /// Lag (shift values backward by offset).
    ///
    /// **Note:** This is a scan-only aggregation. Using it with `GroupBy::aggregate`
    /// will return an error. Use [`GroupByScan`](crate::groupby::GroupByScan) instead.
    Lag { offset: i32 },
    /// Lead (shift values forward by offset).
    ///
    /// **Note:** This is a scan-only aggregation. Using it with `GroupBy::aggregate`
    /// will return an error. Use [`GroupByScan`](crate::groupby::GroupByScan) instead.
    Lead { offset: i32 },
}

/// An owned aggregation object backed by a libcudf `groupby_aggregation`.
pub struct Aggregation {
    pub(crate) inner: UniquePtr<cudf_cxx::aggregation::ffi::OwnedAggregation>,
}

impl Aggregation {
    /// Create an aggregation from the given kind.
    pub fn new(kind: AggregationKind) -> Self {
        use cudf_cxx::aggregation::ffi;

        let inner = match kind {
            AggregationKind::Sum => ffi::agg_sum(),
            AggregationKind::Product => ffi::agg_product(),
            AggregationKind::Min => ffi::agg_min(),
            AggregationKind::Max => ffi::agg_max(),
            AggregationKind::Count => ffi::agg_count(NullHandling::Include as i32),
            AggregationKind::CountValid => ffi::agg_count(NullHandling::Exclude as i32),
            AggregationKind::Any => ffi::agg_any(),
            AggregationKind::All => ffi::agg_all(),
            AggregationKind::SumOfSquares => ffi::agg_sum_of_squares(),
            AggregationKind::Mean => ffi::agg_mean(),
            AggregationKind::Median => ffi::agg_median(),
            AggregationKind::Variance { ddof } => ffi::agg_variance(ddof),
            AggregationKind::Std { ddof } => ffi::agg_std(ddof),
            AggregationKind::Nunique => ffi::agg_nunique(NullHandling::Exclude as i32),
            AggregationKind::NthElement { n } => {
                ffi::agg_nth_element(n, NullHandling::Include as i32)
            }
            AggregationKind::CollectList => ffi::agg_collect_list(NullHandling::Include as i32),
            AggregationKind::CollectSet => ffi::agg_collect_set(NullHandling::Include as i32),
            AggregationKind::Argmax => ffi::agg_argmax(),
            AggregationKind::Argmin => ffi::agg_argmin(),
            AggregationKind::RowNumber => ffi::agg_row_number(),
            AggregationKind::Quantile { q } => ffi::agg_quantile(q),
            AggregationKind::Lag { offset } => ffi::agg_lag(offset),
            AggregationKind::Lead { offset } => ffi::agg_lead(offset),
        };

        Self { inner }
    }
}
