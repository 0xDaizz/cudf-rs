#include "join_shim.h"
#include <cudf/join/join.hpp>
#include <cudf/table/table.hpp>
#include <cudf/column/column_factories.hpp>

namespace cudf_shims {

namespace {

/// Package a pair of gather map device_uvectors into a 2-column OwnedTable.
std::unique_ptr<OwnedTable> package_gather_maps(
    std::unique_ptr<rmm::device_uvector<cudf::size_type>> left_map,
    std::unique_ptr<rmm::device_uvector<cudf::size_type>> right_map)
{
    auto left_col = std::make_unique<cudf::column>(
        cudf::data_type{cudf::type_id::INT32},
        static_cast<cudf::size_type>(left_map->size()),
        left_map->release(),
        rmm::device_buffer{}, 0);

    auto right_col = std::make_unique<cudf::column>(
        cudf::data_type{cudf::type_id::INT32},
        static_cast<cudf::size_type>(right_map->size()),
        right_map->release(),
        rmm::device_buffer{}, 0);

    std::vector<std::unique_ptr<cudf::column>> cols;
    cols.push_back(std::move(left_col));
    cols.push_back(std::move(right_col));

    auto table = std::make_unique<cudf::table>(std::move(cols));
    return std::make_unique<OwnedTable>(std::move(table));
}

} // anonymous namespace

std::unique_ptr<OwnedTable> inner_join(
    const OwnedTable& left_keys, const OwnedTable& right_keys)
{
    auto [left_map, right_map] = cudf::inner_join(
        left_keys.view(), right_keys.view());
    return package_gather_maps(std::move(left_map), std::move(right_map));
}

std::unique_ptr<OwnedTable> left_join(
    const OwnedTable& left_keys, const OwnedTable& right_keys)
{
    auto [left_map, right_map] = cudf::left_join(
        left_keys.view(), right_keys.view());
    return package_gather_maps(std::move(left_map), std::move(right_map));
}

std::unique_ptr<OwnedTable> full_join(
    const OwnedTable& left_keys, const OwnedTable& right_keys)
{
    auto [left_map, right_map] = cudf::full_join(
        left_keys.view(), right_keys.view());
    return package_gather_maps(std::move(left_map), std::move(right_map));
}

std::unique_ptr<OwnedTable> cross_join(
    const OwnedTable& left, const OwnedTable& right)
{
    auto result = cudf::cross_join(left.view(), right.view());
    return std::make_unique<OwnedTable>(std::move(result));
}

} // namespace cudf_shims
