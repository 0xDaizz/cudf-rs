#include "stream_compaction_shim.h"
#include <cudf/stream_compaction.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <stdexcept>

namespace cudf_shims {

namespace {

/// Convert i32 slice to vector of cudf::size_type (column indices).
std::vector<cudf::size_type> to_size_type_vec(rust::Slice<const int32_t> indices) {
    std::vector<cudf::size_type> result;
    result.reserve(indices.size());
    for (auto v : indices) {
        result.push_back(static_cast<cudf::size_type>(v));
    }
    return result;
}

/// Convert i32 to cudf::duplicate_keep_option.
cudf::duplicate_keep_option to_keep_option(int32_t keep) {
    switch (keep) {
        case 0: return cudf::duplicate_keep_option::KEEP_FIRST;
        case 1: return cudf::duplicate_keep_option::KEEP_LAST;
        case 2: return cudf::duplicate_keep_option::KEEP_ANY;
        case 3: return cudf::duplicate_keep_option::KEEP_NONE;
        default: throw std::runtime_error("Invalid keep option: " + std::to_string(keep));
    }
}

/// Convert i32 to cudf::null_equality.
cudf::null_equality to_null_equality(int32_t eq) {
    return eq == 0 ? cudf::null_equality::EQUAL : cudf::null_equality::UNEQUAL;
}

} // anonymous namespace

std::unique_ptr<OwnedTable> drop_nulls_table(
    const OwnedTable& table,
    rust::Slice<const int32_t> keys,
    int32_t threshold)
{
    auto key_vec = to_size_type_vec(keys);
    auto result = cudf::drop_nulls(
        table.view(),
        key_vec,
        static_cast<cudf::size_type>(threshold));
    return std::make_unique<OwnedTable>(std::move(result));
}

std::unique_ptr<OwnedColumn> drop_nulls_column(
    const OwnedColumn& col)
{
    // Build a single-column table, drop nulls, extract the column
    auto view = col.view();
    std::vector<cudf::column_view> views = {view};
    cudf::table_view tv(views);
    std::vector<cudf::size_type> key_vec = {0};

    auto result = cudf::drop_nulls(tv, key_vec, 1);
    auto columns = result->release();
    return std::make_unique<OwnedColumn>(std::move(columns[0]));
}

std::unique_ptr<OwnedTable> drop_nans(
    const OwnedTable& table,
    rust::Slice<const int32_t> keys)
{
    auto key_vec = to_size_type_vec(keys);
    auto result = cudf::drop_nans(table.view(), key_vec);
    return std::make_unique<OwnedTable>(std::move(result));
}

std::unique_ptr<OwnedTable> apply_boolean_mask(
    const OwnedTable& table,
    const OwnedColumn& boolean_mask)
{
    auto result = cudf::apply_boolean_mask(table.view(), boolean_mask.view());
    return std::make_unique<OwnedTable>(std::move(result));
}

std::unique_ptr<OwnedTable> unique(
    const OwnedTable& table,
    rust::Slice<const int32_t> keys,
    int32_t keep,
    int32_t null_equality)
{
    auto key_vec = to_size_type_vec(keys);
    auto result = cudf::unique(
        table.view(),
        key_vec,
        to_keep_option(keep),
        to_null_equality(null_equality));
    return std::make_unique<OwnedTable>(std::move(result));
}

std::unique_ptr<OwnedTable> distinct(
    const OwnedTable& table,
    rust::Slice<const int32_t> keys,
    int32_t keep,
    int32_t null_equality)
{
    auto key_vec = to_size_type_vec(keys);
    auto result = cudf::distinct(
        table.view(),
        key_vec,
        to_keep_option(keep),
        to_null_equality(null_equality));
    return std::make_unique<OwnedTable>(std::move(result));
}

int32_t distinct_count_column(
    const OwnedColumn& col,
    int32_t null_handling,
    int32_t nan_handling)
{
    auto nh = null_handling == 0 ? cudf::null_policy::INCLUDE : cudf::null_policy::EXCLUDE;
    auto nanp = nan_handling == 0 ? cudf::nan_policy::NAN_IS_VALID : cudf::nan_policy::NAN_IS_NULL;
    return cudf::distinct_count(col.view(), nh, nanp);
}

std::unique_ptr<OwnedColumn> distinct_indices(
    const OwnedTable& table,
    int32_t keep,
    int32_t null_equality)
{
    auto result = cudf::distinct_indices(
        table.view(),
        to_keep_option(keep),
        to_null_equality(null_equality));
    return std::make_unique<OwnedColumn>(std::move(result));
}

std::unique_ptr<OwnedTable> stable_distinct(
    const OwnedTable& table,
    rust::Slice<const int32_t> keys,
    int32_t keep,
    int32_t null_equality)
{
    auto key_vec = to_size_type_vec(keys);
    auto result = cudf::stable_distinct(
        table.view(),
        key_vec,
        to_keep_option(keep),
        to_null_equality(null_equality));
    return std::make_unique<OwnedTable>(std::move(result));
}

int32_t unique_count_column(
    const OwnedColumn& col,
    int32_t null_handling,
    int32_t nan_handling)
{
    auto nh = null_handling == 0 ? cudf::null_policy::INCLUDE : cudf::null_policy::EXCLUDE;
    auto nanp = nan_handling == 0 ? cudf::nan_policy::NAN_IS_VALID : cudf::nan_policy::NAN_IS_NULL;
    return cudf::unique_count(col.view(), nh, nanp);
}

int32_t unique_count_table(
    const OwnedTable& table,
    int32_t null_equality)
{
    return cudf::unique_count(table.view(), to_null_equality(null_equality));
}

int32_t distinct_count_table(
    const OwnedTable& table,
    int32_t null_equality)
{
    return cudf::distinct_count(table.view(), to_null_equality(null_equality));
}

} // namespace cudf_shims
