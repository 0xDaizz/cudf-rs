#include "sorting_shim.h"
#include <cudf/sorting.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <stdexcept>

namespace cudf_shims {

namespace {

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

std::unique_ptr<OwnedColumn> sorted_order(
    const OwnedTable& table,
    rust::Slice<const int32_t> column_order,
    rust::Slice<const int32_t> null_order)
{
    auto col_order = to_column_order(column_order);
    auto nul_order = to_null_order(null_order);

    auto result = cudf::sorted_order(
        table.view(),
        col_order,
        nul_order);

    return std::make_unique<OwnedColumn>(std::move(result));
}

std::unique_ptr<OwnedTable> sort(
    const OwnedTable& table,
    rust::Slice<const int32_t> column_order,
    rust::Slice<const int32_t> null_order)
{
    auto col_order = to_column_order(column_order);
    auto nul_order = to_null_order(null_order);

    auto result = cudf::sort(
        table.view(),
        col_order,
        nul_order);

    return std::make_unique<OwnedTable>(std::move(result));
}

std::unique_ptr<OwnedTable> sort_by_key(
    const OwnedTable& values,
    const OwnedTable& keys,
    rust::Slice<const int32_t> column_order,
    rust::Slice<const int32_t> null_order)
{
    auto col_order = to_column_order(column_order);
    auto nul_order = to_null_order(null_order);

    auto result = cudf::sort_by_key(
        values.view(),
        keys.view(),
        col_order,
        nul_order);

    return std::make_unique<OwnedTable>(std::move(result));
}

std::unique_ptr<OwnedTable> stable_sort_by_key(
    const OwnedTable& values,
    const OwnedTable& keys,
    rust::Slice<const int32_t> column_order,
    rust::Slice<const int32_t> null_order)
{
    auto col_order = to_column_order(column_order);
    auto nul_order = to_null_order(null_order);

    auto result = cudf::stable_sort_by_key(
        values.view(),
        keys.view(),
        col_order,
        nul_order);

    return std::make_unique<OwnedTable>(std::move(result));
}

std::unique_ptr<OwnedColumn> rank(
    const OwnedColumn& col,
    int32_t method,
    int32_t column_order,
    int32_t null_order,
    int32_t null_handling,
    bool percentage)
{
    auto m = static_cast<cudf::rank_method>(method);
    auto co = column_order == 0 ? cudf::order::ASCENDING : cudf::order::DESCENDING;
    auto no = null_order == 0 ? cudf::null_order::BEFORE : cudf::null_order::AFTER;
    auto nh = null_handling == 0 ? cudf::null_policy::INCLUDE : cudf::null_policy::EXCLUDE;

    auto result = cudf::rank(
        col.view(),
        m,
        co,
        nh,
        no,
        percentage);

    return std::make_unique<OwnedColumn>(std::move(result));
}

bool is_sorted(
    const OwnedTable& table,
    rust::Slice<const int32_t> column_order,
    rust::Slice<const int32_t> null_order)
{
    auto col_order = to_column_order(column_order);
    auto nul_order = to_null_order(null_order);

    return cudf::is_sorted(
        table.view(),
        col_order,
        nul_order);
}

std::unique_ptr<OwnedColumn> stable_sorted_order(
    const OwnedTable& table,
    rust::Slice<const int32_t> column_order,
    rust::Slice<const int32_t> null_order)
{
    auto col_order = to_column_order(column_order);
    auto nul_order = to_null_order(null_order);

    auto result = cudf::stable_sorted_order(
        table.view(),
        col_order,
        nul_order);

    return std::make_unique<OwnedColumn>(std::move(result));
}

std::unique_ptr<OwnedTable> stable_sort(
    const OwnedTable& table,
    rust::Slice<const int32_t> column_order,
    rust::Slice<const int32_t> null_order)
{
    auto col_order = to_column_order(column_order);
    auto nul_order = to_null_order(null_order);

    auto result = cudf::stable_sort(
        table.view(),
        col_order,
        nul_order);

    return std::make_unique<OwnedTable>(std::move(result));
}

std::unique_ptr<OwnedColumn> top_k(
    const OwnedColumn& col,
    int32_t k,
    int32_t order)
{
    auto ord = order == 0 ? cudf::order::ASCENDING : cudf::order::DESCENDING;

    auto result = cudf::top_k(
        col.view(),
        k,
        ord);

    return std::make_unique<OwnedColumn>(std::move(result));
}

// ── Segmented Sorting ─────────────────────────────────────────

std::unique_ptr<OwnedColumn> segmented_sorted_order(
    const OwnedTable& table,
    const OwnedColumn& segment_offsets,
    rust::Slice<const int32_t> column_order,
    rust::Slice<const int32_t> null_order)
{
    auto col_order = to_column_order(column_order);
    auto nul_order = to_null_order(null_order);

    auto result = cudf::segmented_sorted_order(
        table.view(),
        segment_offsets.view(),
        col_order,
        nul_order);

    return std::make_unique<OwnedColumn>(std::move(result));
}

std::unique_ptr<OwnedColumn> stable_segmented_sorted_order(
    const OwnedTable& table,
    const OwnedColumn& segment_offsets,
    rust::Slice<const int32_t> column_order,
    rust::Slice<const int32_t> null_order)
{
    auto col_order = to_column_order(column_order);
    auto nul_order = to_null_order(null_order);

    auto result = cudf::stable_segmented_sorted_order(
        table.view(),
        segment_offsets.view(),
        col_order,
        nul_order);

    return std::make_unique<OwnedColumn>(std::move(result));
}

std::unique_ptr<OwnedTable> segmented_sort_by_key(
    const OwnedTable& values,
    const OwnedTable& keys,
    const OwnedColumn& segment_offsets,
    rust::Slice<const int32_t> column_order,
    rust::Slice<const int32_t> null_order)
{
    auto col_order = to_column_order(column_order);
    auto nul_order = to_null_order(null_order);

    auto result = cudf::segmented_sort_by_key(
        values.view(),
        keys.view(),
        segment_offsets.view(),
        col_order,
        nul_order);

    return std::make_unique<OwnedTable>(std::move(result));
}

std::unique_ptr<OwnedTable> stable_segmented_sort_by_key(
    const OwnedTable& values,
    const OwnedTable& keys,
    const OwnedColumn& segment_offsets,
    rust::Slice<const int32_t> column_order,
    rust::Slice<const int32_t> null_order)
{
    auto col_order = to_column_order(column_order);
    auto nul_order = to_null_order(null_order);

    auto result = cudf::stable_segmented_sort_by_key(
        values.view(),
        keys.view(),
        segment_offsets.view(),
        col_order,
        nul_order);

    return std::make_unique<OwnedTable>(std::move(result));
}

} // namespace cudf_shims
