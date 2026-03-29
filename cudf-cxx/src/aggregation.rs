//! Bridge definitions for libcudf aggregation factories.
//!
//! Provides `OwnedAggregation` wrapping `std::unique_ptr<cudf::groupby_aggregation>`,
//! with factory functions for each supported aggregation kind.

#[cxx::bridge(namespace = "cudf_shims")]
pub mod ffi {
    unsafe extern "C++" {
        include!("aggregation_shim.h");

        /// Opaque owning handle wrapping a `cudf::groupby_aggregation`.
        type OwnedAggregation;

        // ── Factory functions ──────────────────────────────────────

        fn agg_sum() -> UniquePtr<OwnedAggregation>;
        fn agg_product() -> UniquePtr<OwnedAggregation>;
        fn agg_min() -> UniquePtr<OwnedAggregation>;
        fn agg_max() -> UniquePtr<OwnedAggregation>;
        fn agg_count(null_handling: i32) -> UniquePtr<OwnedAggregation>;
        fn agg_any() -> UniquePtr<OwnedAggregation>;
        fn agg_all() -> UniquePtr<OwnedAggregation>;
        fn agg_sum_of_squares() -> UniquePtr<OwnedAggregation>;
        fn agg_mean() -> UniquePtr<OwnedAggregation>;
        fn agg_median() -> UniquePtr<OwnedAggregation>;
        fn agg_variance(ddof: i32) -> UniquePtr<OwnedAggregation>;
        fn agg_std(ddof: i32) -> UniquePtr<OwnedAggregation>;
        fn agg_nunique(null_handling: i32) -> UniquePtr<OwnedAggregation>;
        fn agg_nth_element(n: i32, null_handling: i32) -> UniquePtr<OwnedAggregation>;
        fn agg_collect_list(null_handling: i32) -> UniquePtr<OwnedAggregation>;
        fn agg_collect_set(null_handling: i32) -> UniquePtr<OwnedAggregation>;
        fn agg_argmax() -> UniquePtr<OwnedAggregation>;
        fn agg_argmin() -> UniquePtr<OwnedAggregation>;
        fn agg_row_number() -> UniquePtr<OwnedAggregation>;
        fn agg_quantile(q: f64) -> UniquePtr<OwnedAggregation>;
        fn agg_lag(offset: i32) -> UniquePtr<OwnedAggregation>;
        fn agg_lead(offset: i32) -> UniquePtr<OwnedAggregation>;
    }
}
