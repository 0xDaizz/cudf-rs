#include "quantiles_shim.h"
#include <cudf/quantiles.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <stdexcept>

namespace cudf_shims {

namespace {

/// Convert an integer interpolation type to cudf::interpolation.
cudf::interpolation to_interpolation(int32_t interp) {
    switch (interp) {
        case 0: return cudf::interpolation::LINEAR;
        case 1: return cudf::interpolation::LOWER;
        case 2: return cudf::interpolation::HIGHER;
        case 3: return cudf::interpolation::MIDPOINT;
        case 4: return cudf::interpolation::NEAREST;
        default:
            throw std::runtime_error("unknown interpolation kind: " + std::to_string(interp));
    }
}

/// Convert a flat i32 slice to a vector of cudf::order.
std::vector<cudf::order> to_column_order(rust::Slice<const int32_t> order) {
    std::vector<cudf::order> result;
    result.reserve(order.size());
    for (auto v : order) {
        result.push_back(v == 0 ? cudf::order::ASCENDING : cudf::order::DESCENDING);
    }
    return result;
}

/// Convert a flat i32 slice to a vector of cudf::null_order.
std::vector<cudf::null_order> to_null_order(rust::Slice<const int32_t> order) {
    std::vector<cudf::null_order> result;
    result.reserve(order.size());
    for (auto v : order) {
        result.push_back(v == 0 ? cudf::null_order::BEFORE : cudf::null_order::AFTER);
    }
    return result;
}

} // anonymous namespace

std::unique_ptr<OwnedColumn> quantile(
    const OwnedColumn& col,
    rust::Slice<const double> q,
    int32_t interp)
{
    std::vector<double> quantiles(q.begin(), q.end());
    auto interpolation = to_interpolation(interp);

    auto result = cudf::quantile(
        col.view(),
        quantiles,
        interpolation);

    return std::make_unique<OwnedColumn>(std::move(result));
}

std::unique_ptr<OwnedTable> quantiles_table(
    const OwnedTable& table,
    rust::Slice<const double> q,
    int32_t interp,
    bool is_input_sorted,
    rust::Slice<const int32_t> orders,
    rust::Slice<const int32_t> null_orders)
{
    std::vector<double> quantiles(q.begin(), q.end());
    auto interpolation = to_interpolation(interp);
    auto col_order = to_column_order(orders);
    auto nul_order = to_null_order(null_orders);

    auto sorted = is_input_sorted
        ? cudf::sorted::YES
        : cudf::sorted::NO;

    auto result = cudf::quantiles(
        table.view(),
        quantiles,
        interpolation,
        sorted,
        col_order,
        nul_order);

    return std::make_unique<OwnedTable>(std::move(result));
}

std::unique_ptr<OwnedColumn> percentile_approx(
    const OwnedColumn& tdigest_col,
    rust::Slice<const double> percentiles)
{
    std::vector<double> pcts(percentiles.begin(), percentiles.end());

    auto pct_col = cudf::make_fixed_width_column(
        cudf::data_type{cudf::type_id::FLOAT64},
        static_cast<cudf::size_type>(pcts.size()));

    // Copy percentiles to device column
    auto stream = cudf::get_default_stream();
    CUDF_CUDA_TRY(cudaMemcpyAsync(
        pct_col->mutable_view().data<double>(),
        pcts.data(),
        pcts.size() * sizeof(double),
        cudaMemcpyHostToDevice,
        stream.value()));
    stream.synchronize();

    auto result = cudf::percentile_approx(
        tdigest_col.view(),
        pct_col->view());

    return std::make_unique<OwnedColumn>(std::move(result));
}

} // namespace cudf_shims
