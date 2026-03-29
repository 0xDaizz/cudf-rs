#pragma once

#include <cudf/aggregation.hpp>
#include <cudf/types.hpp>
#include <memory>
#include "rust/cxx.h"

namespace cudf_shims {

/// Owning wrapper around a cudf::groupby_aggregation.
struct OwnedAggregation {
    std::unique_ptr<cudf::groupby_aggregation> inner;

    explicit OwnedAggregation(std::unique_ptr<cudf::groupby_aggregation> agg)
        : inner(std::move(agg)) {}
};

// ── Factory functions ──────────────────────────────────────────

std::unique_ptr<OwnedAggregation> agg_sum();
std::unique_ptr<OwnedAggregation> agg_product();
std::unique_ptr<OwnedAggregation> agg_min();
std::unique_ptr<OwnedAggregation> agg_max();
std::unique_ptr<OwnedAggregation> agg_count(int32_t null_handling);
std::unique_ptr<OwnedAggregation> agg_any();
std::unique_ptr<OwnedAggregation> agg_all();
std::unique_ptr<OwnedAggregation> agg_sum_of_squares();
std::unique_ptr<OwnedAggregation> agg_mean();
std::unique_ptr<OwnedAggregation> agg_median();
std::unique_ptr<OwnedAggregation> agg_variance(int32_t ddof);
std::unique_ptr<OwnedAggregation> agg_std(int32_t ddof);
std::unique_ptr<OwnedAggregation> agg_nunique(int32_t null_handling);
std::unique_ptr<OwnedAggregation> agg_nth_element(int32_t n, int32_t null_handling);
std::unique_ptr<OwnedAggregation> agg_collect_list(int32_t null_handling);
std::unique_ptr<OwnedAggregation> agg_collect_set(int32_t null_handling);
std::unique_ptr<OwnedAggregation> agg_argmax();
std::unique_ptr<OwnedAggregation> agg_argmin();
std::unique_ptr<OwnedAggregation> agg_row_number();
std::unique_ptr<OwnedAggregation> agg_quantile(double q);
std::unique_ptr<OwnedAggregation> agg_lag(int32_t offset);
std::unique_ptr<OwnedAggregation> agg_lead(int32_t offset);

} // namespace cudf_shims
