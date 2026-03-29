#include "search_shim.h"
#include "scalar_shim.h"
#include <cudf/search.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <vector>

namespace cudf_shims {

namespace {

std::vector<cudf::order> to_column_order(rust::Slice<const int32_t> order) {
    std::vector<cudf::order> result;
    result.reserve(order.size());
    for (auto v : order) {
        result.push_back(v == 0 ? cudf::order::ASCENDING : cudf::order::DESCENDING);
    }
    return result;
}

std::vector<cudf::null_order> to_null_order(rust::Slice<const int32_t> order) {
    std::vector<cudf::null_order> result;
    result.reserve(order.size());
    for (auto v : order) {
        result.push_back(v == 0 ? cudf::null_order::AFTER : cudf::null_order::BEFORE);
    }
    return result;
}

} // anonymous namespace

std::unique_ptr<OwnedColumn> lower_bound(
    const OwnedTable& table,
    const OwnedTable& values,
    rust::Slice<const int32_t> orders,
    rust::Slice<const int32_t> null_orders)
{
    auto col_order = to_column_order(orders);
    auto nul_order = to_null_order(null_orders);

    auto result = cudf::lower_bound(
        table.view(),
        values.view(),
        col_order,
        nul_order);

    return std::make_unique<OwnedColumn>(std::move(result));
}

std::unique_ptr<OwnedColumn> upper_bound(
    const OwnedTable& table,
    const OwnedTable& values,
    rust::Slice<const int32_t> orders,
    rust::Slice<const int32_t> null_orders)
{
    auto col_order = to_column_order(orders);
    auto nul_order = to_null_order(null_orders);

    auto result = cudf::upper_bound(
        table.view(),
        values.view(),
        col_order,
        nul_order);

    return std::make_unique<OwnedColumn>(std::move(result));
}

std::unique_ptr<OwnedColumn> contains_column(
    const OwnedColumn& haystack,
    const OwnedColumn& needles)
{
    auto result = cudf::contains(haystack.view(), needles.view());
    return std::make_unique<OwnedColumn>(std::move(result));
}

bool contains_scalar(
    const OwnedColumn& haystack,
    const OwnedScalar& needle)
{
    return cudf::contains(haystack.view(), *needle.inner);
}

} // namespace cudf_shims
