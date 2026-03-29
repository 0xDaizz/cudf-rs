#include "copying_shim.h"
#include <cudf/copying.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <stdexcept>

namespace cudf_shims {

std::unique_ptr<OwnedTable> gather(
    const OwnedTable& table,
    const OwnedColumn& gather_map,
    int32_t bounds_policy)
{
    auto policy = bounds_policy == 0
        ? cudf::out_of_bounds_policy::DONT_CHECK
        : cudf::out_of_bounds_policy::NULLIFY;

    auto result = cudf::gather(
        table.view(),
        gather_map.view(),
        policy);

    return std::make_unique<OwnedTable>(std::move(result));
}

std::unique_ptr<OwnedTable> scatter(
    const OwnedTable& source,
    const OwnedColumn& scatter_map,
    const OwnedTable& target)
{
    auto result = cudf::scatter(
        source.view(),
        scatter_map.view(),
        target.view());

    return std::make_unique<OwnedTable>(std::move(result));
}

std::unique_ptr<OwnedColumn> copy_if_else(
    const OwnedColumn& lhs,
    const OwnedColumn& rhs,
    const OwnedColumn& boolean_mask)
{
    auto result = cudf::copy_if_else(
        lhs.view(),
        rhs.view(),
        boolean_mask.view());

    return std::make_unique<OwnedColumn>(std::move(result));
}

std::unique_ptr<OwnedTable> slice_table(
    const OwnedTable& table,
    int32_t begin,
    int32_t end)
{
    // cudf::slice returns views; we deep-copy to return owned data.
    std::vector<cudf::size_type> indices = {begin, end};
    auto views = cudf::slice(table.view(), indices);

    if (views.empty()) {
        throw std::runtime_error("slice returned no results");
    }

    // Deep-copy the view into an owned table
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();

    auto owned = std::make_unique<cudf::table>(views[0], stream, mr);
    return std::make_unique<OwnedTable>(std::move(owned));
}

std::unique_ptr<SplitResult> split_table_all(
    const OwnedTable& table, rust::Slice<const int32_t> splits)
{
    std::vector<cudf::size_type> split_vec(splits.begin(), splits.end());
    auto views = cudf::split(table.view(), split_vec);
    auto stream = cudf::get_default_stream();
    auto mr = cudf::get_current_device_resource_ref();

    auto result = std::make_unique<SplitResult>();
    for (auto& view : views) {
        auto owned = std::make_unique<cudf::table>(view, stream, mr);
        result->parts.push_back(std::make_unique<OwnedTable>(std::move(owned)));
    }
    return result;
}

int32_t split_result_count(const SplitResult& result) {
    return static_cast<int32_t>(result.parts.size());
}

std::unique_ptr<OwnedTable> split_result_get(SplitResult& result, int32_t index) {
    if (index < 0 || static_cast<size_t>(index) >= result.parts.size()) {
        throw std::runtime_error("split result index out of bounds");
    }
    if (!result.parts[index]) {
        throw std::runtime_error("split result part already consumed");
    }
    return std::move(result.parts[index]);
}

std::unique_ptr<OwnedColumn> empty_like(const OwnedColumn& col) {
    auto result = cudf::empty_like(col.view());
    return std::make_unique<OwnedColumn>(std::move(result));
}

std::unique_ptr<OwnedColumn> allocate_like(
    const OwnedColumn& col,
    int32_t mask_policy)
{
    auto policy = static_cast<cudf::mask_allocation_policy>(mask_policy);
    auto result = cudf::allocate_like(col.view(), policy);
    return std::make_unique<OwnedColumn>(std::move(result));
}

void copy_range(
    const OwnedColumn& source,
    OwnedColumn& target,
    int32_t source_begin,
    int32_t source_end,
    int32_t target_begin)
{
    auto mut_view = target.mutable_view();
    cudf::copy_range_in_place(
        source.view(),
        mut_view,
        source_begin,
        source_end,
        target_begin);
}

} // namespace cudf_shims
