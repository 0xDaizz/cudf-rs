#include "reduction_shim.h"
#include <cudf/reduction.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <stdexcept>

namespace cudf_shims {

namespace {

/// Create a reduce aggregation from an integer kind.
std::unique_ptr<cudf::reduce_aggregation> make_reduce_agg(int32_t agg_kind) {
    switch (agg_kind) {
        case 0: return cudf::make_sum_aggregation<cudf::reduce_aggregation>();
        case 1: return cudf::make_product_aggregation<cudf::reduce_aggregation>();
        case 2: return cudf::make_min_aggregation<cudf::reduce_aggregation>();
        case 3: return cudf::make_max_aggregation<cudf::reduce_aggregation>();
        case 4: return cudf::make_sum_of_squares_aggregation<cudf::reduce_aggregation>();
        case 5: return cudf::make_mean_aggregation<cudf::reduce_aggregation>();
        case 6: return cudf::make_variance_aggregation<cudf::reduce_aggregation>();
        case 7: return cudf::make_std_aggregation<cudf::reduce_aggregation>();
        case 8: return cudf::make_any_aggregation<cudf::reduce_aggregation>();
        case 9: return cudf::make_all_aggregation<cudf::reduce_aggregation>();
        case 10: return cudf::make_median_aggregation<cudf::reduce_aggregation>();
        default:
            throw std::runtime_error("unknown reduce aggregation kind: " + std::to_string(agg_kind));
    }
}

/// Create a scan aggregation from an integer kind.
std::unique_ptr<cudf::scan_aggregation> make_scan_agg(int32_t agg_kind) {
    switch (agg_kind) {
        case 0: return cudf::make_sum_aggregation<cudf::scan_aggregation>();
        case 1: return cudf::make_product_aggregation<cudf::scan_aggregation>();
        case 2: return cudf::make_min_aggregation<cudf::scan_aggregation>();
        case 3: return cudf::make_max_aggregation<cudf::scan_aggregation>();
        default:
            throw std::runtime_error("unknown scan aggregation kind: " + std::to_string(agg_kind));
    }
}

/// Create a segmented reduce aggregation from an integer kind.
std::unique_ptr<cudf::segmented_reduce_aggregation> make_segmented_reduce_agg(int32_t agg_kind) {
    switch (agg_kind) {
        case 0: return cudf::make_sum_aggregation<cudf::segmented_reduce_aggregation>();
        case 1: return cudf::make_product_aggregation<cudf::segmented_reduce_aggregation>();
        case 2: return cudf::make_min_aggregation<cudf::segmented_reduce_aggregation>();
        case 3: return cudf::make_max_aggregation<cudf::segmented_reduce_aggregation>();
        case 4: return cudf::make_sum_of_squares_aggregation<cudf::segmented_reduce_aggregation>();
        case 5: return cudf::make_mean_aggregation<cudf::segmented_reduce_aggregation>();
        case 6: return cudf::make_variance_aggregation<cudf::segmented_reduce_aggregation>();
        case 7: return cudf::make_std_aggregation<cudf::segmented_reduce_aggregation>();
        case 8: return cudf::make_any_aggregation<cudf::segmented_reduce_aggregation>();
        case 9: return cudf::make_all_aggregation<cudf::segmented_reduce_aggregation>();
        case 10: return cudf::make_median_aggregation<cudf::segmented_reduce_aggregation>();
        default:
            throw std::runtime_error("unknown segmented reduce aggregation kind: " + std::to_string(agg_kind));
    }
}

} // anonymous namespace

std::unique_ptr<OwnedScalar> reduce(
    const OwnedColumn& col,
    int32_t agg_kind,
    int32_t output_type_id)
{
    auto agg = make_reduce_agg(agg_kind);
    auto out_type = cudf::data_type{static_cast<cudf::type_id>(output_type_id)};

    auto result = cudf::reduce(
        col.view(),
        *agg,
        out_type);

    return std::make_unique<OwnedScalar>(std::move(result));
}

std::unique_ptr<OwnedColumn> scan(
    const OwnedColumn& col,
    int32_t agg_kind,
    bool inclusive)
{
    auto agg = make_scan_agg(agg_kind);
    auto scan_type = inclusive
        ? cudf::scan_type::INCLUSIVE
        : cudf::scan_type::EXCLUSIVE;

    auto result = cudf::scan(
        col.view(),
        *agg,
        scan_type);

    return std::make_unique<OwnedColumn>(std::move(result));
}

std::unique_ptr<OwnedColumn> segmented_reduce(
    const OwnedColumn& col,
    const OwnedColumn& offsets,
    int32_t agg_kind,
    int32_t output_type_id,
    bool include_nulls)
{
    auto agg = make_segmented_reduce_agg(agg_kind);
    auto out_type = cudf::data_type{static_cast<cudf::type_id>(output_type_id)};
    auto null_policy = include_nulls
        ? cudf::null_policy::INCLUDE
        : cudf::null_policy::EXCLUDE;

    auto result = cudf::segmented_reduce(
        col.view(),
        offsets.view(),
        *agg,
        out_type,
        null_policy);

    return std::make_unique<OwnedColumn>(std::move(result));
}

std::unique_ptr<MinMaxResult> minmax(const OwnedColumn& col) {
    auto [min_scalar, max_scalar] = cudf::minmax(col.view());
    auto result = std::make_unique<MinMaxResult>();
    result->min_val = std::make_unique<OwnedScalar>(std::move(min_scalar));
    result->max_val = std::make_unique<OwnedScalar>(std::move(max_scalar));
    return result;
}

std::unique_ptr<OwnedScalar> minmax_take_min(MinMaxResult& result) {
    return std::move(result.min_val);
}

std::unique_ptr<OwnedScalar> minmax_take_max(MinMaxResult& result) {
    return std::move(result.max_val);
}

std::unique_ptr<OwnedScalar> reduce_with_init(
    const OwnedColumn& col,
    int32_t agg_kind,
    int32_t output_type_id,
    const OwnedScalar& init)
{
    auto agg = make_reduce_agg(agg_kind);
    auto out_type = cudf::data_type{static_cast<cudf::type_id>(output_type_id)};

    auto result = cudf::reduce(
        col.view(),
        *agg,
        out_type,
        std::optional<std::reference_wrapper<cudf::scalar const>>{*init.inner});

    return std::make_unique<OwnedScalar>(std::move(result));
}

bool is_valid_reduction_aggregation(int32_t source_type_id, int32_t agg_kind)
{
    auto src_type = cudf::data_type{static_cast<cudf::type_id>(source_type_id)};
    // Use make_reduce_agg to get the correct aggregation (same mapping as reduce()),
    // then extract its Kind — avoids assuming ReduceOp numbering == aggregation::Kind.
    auto agg = make_reduce_agg(agg_kind);
    return cudf::reduction::is_valid_aggregation(src_type, agg->kind);
}


std::unique_ptr<OwnedScalar> reduce_var_with_ddof(
    const OwnedColumn& col,
    int32_t ddof,
    int32_t output_type_id)
{
    auto agg = cudf::make_variance_aggregation<cudf::reduce_aggregation>(
        static_cast<cudf::size_type>(ddof));
    auto out_type = cudf::data_type{static_cast<cudf::type_id>(output_type_id)};

    auto result = cudf::reduce(col.view(), *agg, out_type);
    return std::make_unique<OwnedScalar>(std::move(result));
}

std::unique_ptr<OwnedScalar> reduce_std_with_ddof(
    const OwnedColumn& col,
    int32_t ddof,
    int32_t output_type_id)
{
    auto agg = cudf::make_std_aggregation<cudf::reduce_aggregation>(
        static_cast<cudf::size_type>(ddof));
    auto out_type = cudf::data_type{static_cast<cudf::type_id>(output_type_id)};

    auto result = cudf::reduce(col.view(), *agg, out_type);
    return std::make_unique<OwnedScalar>(std::move(result));
}

} // namespace cudf_shims
