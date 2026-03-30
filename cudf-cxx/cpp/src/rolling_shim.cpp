#include "rolling_shim.h"
#include <cudf/rolling.hpp>
#include <cudf/aggregation.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <stdexcept>

namespace cudf_shims {

namespace {

/// Create a rolling aggregation from an integer kind.
std::unique_ptr<cudf::rolling_aggregation> make_rolling_agg(int32_t agg_kind) {
    switch (agg_kind) {
        case 0: return cudf::make_sum_aggregation<cudf::rolling_aggregation>();
        case 1: return cudf::make_min_aggregation<cudf::rolling_aggregation>();
        case 2: return cudf::make_max_aggregation<cudf::rolling_aggregation>();
        case 3: return cudf::make_count_aggregation<cudf::rolling_aggregation>();
        case 4: return cudf::make_mean_aggregation<cudf::rolling_aggregation>();
        case 5: return cudf::make_collect_list_aggregation<cudf::rolling_aggregation>();
        case 6: return cudf::make_row_number_aggregation<cudf::rolling_aggregation>();
        case 7: return cudf::make_lead_aggregation<cudf::rolling_aggregation>(1);
        case 8: return cudf::make_lag_aggregation<cudf::rolling_aggregation>(1);
        default:
            throw std::runtime_error("unknown rolling aggregation kind: " + std::to_string(agg_kind));
    }
}

} // anonymous namespace

std::unique_ptr<OwnedColumn> rolling_window(
    const OwnedColumn& col,
    int32_t preceding,
    int32_t following,
    int32_t min_periods,
    int32_t agg_kind)
{
    auto agg = make_rolling_agg(agg_kind);

    auto result = cudf::rolling_window(
        col.view(),
        preceding,
        following,
        min_periods,
        *agg);

    return std::make_unique<OwnedColumn>(std::move(result));
}

std::unique_ptr<OwnedColumn> grouped_rolling_window(
    const OwnedTable& group_keys,
    const OwnedColumn& input,
    int32_t preceding,
    int32_t following,
    int32_t min_periods,
    int32_t agg_kind)
{
    auto agg = make_rolling_agg(agg_kind);

    auto result = cudf::grouped_rolling_window(
        group_keys.view(),
        input.view(),
        preceding,
        following,
        min_periods,
        *agg);

    return std::make_unique<OwnedColumn>(std::move(result));
}

std::unique_ptr<OwnedColumn> rolling_window_variable(
    const OwnedColumn& col,
    const OwnedColumn& preceding_col,
    const OwnedColumn& following_col,
    int32_t min_periods,
    int32_t agg_kind)
{
    auto agg = make_rolling_agg(agg_kind);

    auto result = cudf::rolling_window(
        col.view(),
        preceding_col.view(),
        following_col.view(),
        min_periods,
        *agg);

    return std::make_unique<OwnedColumn>(std::move(result));
}

bool is_valid_rolling_aggregation(int32_t source_type_id, int32_t agg_kind)
{
    auto src_type = cudf::data_type{static_cast<cudf::type_id>(source_type_id)};
    // Use make_rolling_agg to get the correct aggregation (same mapping as rolling_window()),
    // then extract its Kind — avoids assuming RollingAgg numbering == aggregation::Kind.
    auto agg = make_rolling_agg(agg_kind);
    return cudf::is_valid_rolling_aggregation(src_type, agg->kind);
}

} // namespace cudf_shims
