#include "join_shim.h"
#include <cudf/join/join.hpp>
#include <cudf/join/filtered_join.hpp>
#include <cudf/table/table.hpp>
#include <cudf/column/column_factories.hpp>

namespace cudf_shims {

namespace {

/// Package a pair of gather map device_uvectors into a 2-column OwnedTable.
std::unique_ptr<OwnedTable> package_gather_maps(
    std::unique_ptr<rmm::device_uvector<cudf::size_type>> left_map,
    std::unique_ptr<rmm::device_uvector<cudf::size_type>> right_map)
{
    // Capture sizes BEFORE release() — C++ function argument evaluation
    // order is unspecified, so release() could be evaluated before size().
    auto left_size = static_cast<cudf::size_type>(left_map->size());
    auto left_buf = left_map->release();
    auto left_col = std::make_unique<cudf::column>(
        cudf::data_type{cudf::type_id::INT32},
        left_size,
        std::move(left_buf),
        rmm::device_buffer{}, 0);

    auto right_size = static_cast<cudf::size_type>(right_map->size());
    auto right_buf = right_map->release();
    auto right_col = std::make_unique<cudf::column>(
        cudf::data_type{cudf::type_id::INT32},
        right_size,
        std::move(right_buf),
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

namespace {

/// Package a single gather map device_uvector into a 1-column OwnedTable.
std::unique_ptr<OwnedTable> package_single_map(
    std::unique_ptr<rmm::device_uvector<cudf::size_type>> map)
{
    // Capture size BEFORE release() — eval order is unspecified in C++
    auto map_size = static_cast<cudf::size_type>(map->size());
    auto map_buf = map->release();
    auto col = std::make_unique<cudf::column>(
        cudf::data_type{cudf::type_id::INT32},
        map_size,
        std::move(map_buf),
        rmm::device_buffer{}, 0);

    std::vector<std::unique_ptr<cudf::column>> cols;
    cols.push_back(std::move(col));

    auto table = std::make_unique<cudf::table>(std::move(cols));
    return std::make_unique<OwnedTable>(std::move(table));
}

} // anonymous namespace

std::unique_ptr<OwnedTable> left_semi_join(
    const OwnedTable& left_keys, const OwnedTable& right_keys)
{
    cudf::filtered_join fj(right_keys.view(), cudf::null_equality::EQUAL, cudf::set_as_build_table::RIGHT, 0.5);
    auto map = fj.semi_join(left_keys.view());
    return package_single_map(std::move(map));
}

std::unique_ptr<OwnedTable> left_anti_join(
    const OwnedTable& left_keys, const OwnedTable& right_keys)
{
    cudf::filtered_join fj(right_keys.view(), cudf::null_equality::EQUAL, cudf::set_as_build_table::RIGHT, 0.5);
    auto map = fj.anti_join(left_keys.view());
    return package_single_map(std::move(map));
}

// ── Hash Join ─────────────────────────────────────────────────

std::unique_ptr<OwnedHashJoin> hash_join_create(const OwnedTable& build) {
    auto hj = std::make_unique<cudf::hash_join>(
        build.view(),
        cudf::null_equality::EQUAL);
    return std::make_unique<OwnedHashJoin>(std::move(hj));
}

std::unique_ptr<OwnedTable> hash_join_inner(
    const OwnedHashJoin& hj, const OwnedTable& probe)
{
    auto [left_map, right_map] = hj.inner->inner_join(probe.view());
    return package_gather_maps(std::move(left_map), std::move(right_map));
}

std::unique_ptr<OwnedTable> hash_join_left(
    const OwnedHashJoin& hj, const OwnedTable& probe)
{
    auto [left_map, right_map] = hj.inner->left_join(probe.view());
    return package_gather_maps(std::move(left_map), std::move(right_map));
}

std::unique_ptr<OwnedTable> hash_join_full(
    const OwnedHashJoin& hj, const OwnedTable& probe)
{
    auto [left_map, right_map] = hj.inner->full_join(probe.view());
    return package_gather_maps(std::move(left_map), std::move(right_map));
}

int64_t hash_join_inner_size(const OwnedHashJoin& hj, const OwnedTable& probe) {
    return static_cast<int64_t>(hj.inner->inner_join_size(probe.view()));
}

int64_t hash_join_left_size(const OwnedHashJoin& hj, const OwnedTable& probe) {
    return static_cast<int64_t>(hj.inner->left_join_size(probe.view()));
}

int64_t hash_join_full_size(const OwnedHashJoin& hj, const OwnedTable& probe) {
    return static_cast<int64_t>(hj.inner->full_join_size(probe.view()));
}

} // namespace cudf_shims
