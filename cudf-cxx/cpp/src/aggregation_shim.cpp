#include "aggregation_shim.h"
#include <cudf/aggregation.hpp>

namespace cudf_shims {

namespace {

/// Helper: downcast a generic aggregation unique_ptr to groupby_aggregation.
/// libcudf's make_*_aggregation<groupby_aggregation>() returns the correct type directly.
template <typename T>
std::unique_ptr<OwnedAggregation> wrap(std::unique_ptr<T> agg) {
    // The aggregation is already a groupby_aggregation due to the template parameter.
    return std::make_unique<OwnedAggregation>(std::move(agg));
}

cudf::null_policy to_null_policy(int32_t v) {
    return v == 0 ? cudf::null_policy::EXCLUDE : cudf::null_policy::INCLUDE;
}

} // anonymous namespace

std::unique_ptr<OwnedAggregation> agg_sum() {
    return wrap(cudf::make_sum_aggregation<cudf::groupby_aggregation>());
}

std::unique_ptr<OwnedAggregation> agg_product() {
    return wrap(cudf::make_product_aggregation<cudf::groupby_aggregation>());
}

std::unique_ptr<OwnedAggregation> agg_min() {
    return wrap(cudf::make_min_aggregation<cudf::groupby_aggregation>());
}

std::unique_ptr<OwnedAggregation> agg_max() {
    return wrap(cudf::make_max_aggregation<cudf::groupby_aggregation>());
}

std::unique_ptr<OwnedAggregation> agg_count(int32_t null_handling) {
    return wrap(cudf::make_count_aggregation<cudf::groupby_aggregation>(
        to_null_policy(null_handling)));
}

std::unique_ptr<OwnedAggregation> agg_any() {
    throw std::runtime_error(
        "any is not supported as a groupby aggregation in this libcudf version. "
        "Use reduce aggregation instead.");
}

std::unique_ptr<OwnedAggregation> agg_all() {
    throw std::runtime_error(
        "all is not supported as a groupby aggregation in this libcudf version. "
        "Use reduce aggregation instead.");
}

std::unique_ptr<OwnedAggregation> agg_sum_of_squares() {
    return wrap(cudf::make_sum_of_squares_aggregation<cudf::groupby_aggregation>());
}

std::unique_ptr<OwnedAggregation> agg_mean() {
    return wrap(cudf::make_mean_aggregation<cudf::groupby_aggregation>());
}

std::unique_ptr<OwnedAggregation> agg_median() {
    return wrap(cudf::make_median_aggregation<cudf::groupby_aggregation>());
}

std::unique_ptr<OwnedAggregation> agg_variance(int32_t ddof) {
    return wrap(cudf::make_variance_aggregation<cudf::groupby_aggregation>(
        static_cast<cudf::size_type>(ddof)));
}

std::unique_ptr<OwnedAggregation> agg_std(int32_t ddof) {
    return wrap(cudf::make_std_aggregation<cudf::groupby_aggregation>(
        static_cast<cudf::size_type>(ddof)));
}

std::unique_ptr<OwnedAggregation> agg_nunique(int32_t null_handling) {
    return wrap(cudf::make_nunique_aggregation<cudf::groupby_aggregation>(
        to_null_policy(null_handling)));
}

std::unique_ptr<OwnedAggregation> agg_nth_element(int32_t n, int32_t null_handling) {
    return wrap(cudf::make_nth_element_aggregation<cudf::groupby_aggregation>(
        static_cast<cudf::size_type>(n),
        to_null_policy(null_handling)));
}

std::unique_ptr<OwnedAggregation> agg_collect_list(int32_t null_handling) {
    return wrap(cudf::make_collect_list_aggregation<cudf::groupby_aggregation>(
        to_null_policy(null_handling)));
}

std::unique_ptr<OwnedAggregation> agg_collect_set(int32_t null_handling) {
    return wrap(cudf::make_collect_set_aggregation<cudf::groupby_aggregation>(
        to_null_policy(null_handling)));
}

std::unique_ptr<OwnedAggregation> agg_argmax() {
    return wrap(cudf::make_argmax_aggregation<cudf::groupby_aggregation>());
}

std::unique_ptr<OwnedAggregation> agg_argmin() {
    return wrap(cudf::make_argmin_aggregation<cudf::groupby_aggregation>());
}

std::unique_ptr<OwnedAggregation> agg_row_number() {
    throw std::runtime_error(
        "row_number is a scan-only aggregation and cannot be used with groupby aggregate. "
        "Use GroupByScan instead.");
}

std::unique_ptr<OwnedAggregation> agg_quantile(double q) {
    return wrap(cudf::make_quantile_aggregation<cudf::groupby_aggregation>(
        {q}, cudf::interpolation::LINEAR));
}

std::unique_ptr<OwnedAggregation> agg_lag(int32_t offset) {
    throw std::runtime_error(
        "lag is a scan-only aggregation and cannot be used with groupby aggregate. "
        "Use GroupByScan instead.");
}

std::unique_ptr<OwnedAggregation> agg_lead(int32_t offset) {
    throw std::runtime_error(
        "lead is a scan-only aggregation and cannot be used with groupby aggregate. "
        "Use GroupByScan instead.");
}

} // namespace cudf_shims
