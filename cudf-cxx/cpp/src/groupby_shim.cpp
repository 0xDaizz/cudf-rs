#include "groupby_shim.h"
#include <cudf/groupby.hpp>
#include <cudf/table/table.hpp>
#include <cudf/column/column.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/replace.hpp>
#include <stdexcept>
#include <utility>

namespace cudf_shims {

namespace {

/// Build aggregation_request vector from the builder's accumulated requests.
/// NOTE: This moves the aggregation unique_ptrs out, so the builder is consumed.
std::vector<cudf::groupby::aggregation_request> build_requests(
    GroupByBuilder& builder,
    const cudf::table_view& values)
{
    // Group requests by column index.
    // libcudf expects one aggregation_request per value column, each containing
    // a vector of aggregations for that column.
    std::map<cudf::size_type, std::vector<std::unique_ptr<cudf::groupby_aggregation>>> grouped;
    for (auto& [col_idx, agg] : builder.requests) {
        grouped[col_idx].push_back(std::move(agg));
    }

    std::vector<cudf::groupby::aggregation_request> result;
    result.reserve(grouped.size());
    for (auto& [col_idx, aggs] : grouped) {
        if (col_idx < 0 || col_idx >= values.num_columns()) {
            throw std::out_of_range(
                "aggregation column index " + std::to_string(col_idx) +
                " out of range [0, " + std::to_string(values.num_columns()) + ")");
        }
        cudf::groupby::aggregation_request req;
        req.values = values.column(col_idx);
        req.aggregations = std::move(aggs);
        result.push_back(std::move(req));
    }

    return result;
}

/// Execute groupby and return (keys_table, values_columns_flattened).
std::pair<std::unique_ptr<cudf::table>, std::vector<std::unique_ptr<cudf::column>>>
execute_impl(GroupByBuilder& builder, const OwnedTable& values)
{
    cudf::groupby::groupby gb(builder.keys_view);
    auto requests = build_requests(builder, values.view());
    auto [keys_result, agg_results] = gb.aggregate(requests);

    // Flatten all result columns from all aggregation results into a single vector.
    std::vector<std::unique_ptr<cudf::column>> value_cols;
    for (auto& result_set : agg_results) {
        for (auto& col : result_set.results) {
            value_cols.push_back(std::move(col));
        }
    }

    return {std::move(keys_result), std::move(value_cols)};
}

} // anonymous namespace

std::unique_ptr<GroupByBuilder> groupby_new(const OwnedTable& keys) {
    return std::make_unique<GroupByBuilder>(keys.view());
}

void groupby_add_request(
    GroupByBuilder& builder,
    int32_t col_idx,
    std::unique_ptr<OwnedAggregation> agg)
{
    builder.requests.emplace_back(
        static_cast<cudf::size_type>(col_idx),
        std::move(agg->inner));
}

std::unique_ptr<OwnedTable> groupby_execute(
    GroupByBuilder& builder,
    const OwnedTable& values)
{
    auto [keys_tbl, value_cols] = execute_impl(builder, values);

    // Combine: key columns + value columns into a single table.
    auto key_cols = keys_tbl->release();
    std::vector<std::unique_ptr<cudf::column>> all_cols;
    all_cols.reserve(key_cols.size() + value_cols.size());
    for (auto& c : key_cols) {
        all_cols.push_back(std::move(c));
    }
    for (auto& c : value_cols) {
        all_cols.push_back(std::move(c));
    }

    auto combined = std::make_unique<cudf::table>(std::move(all_cols));
    return std::make_unique<OwnedTable>(std::move(combined));
}

std::unique_ptr<OwnedTable> groupby_execute_keys(
    GroupByBuilder& builder,
    const OwnedTable& values)
{
    auto [keys_tbl, value_cols] = execute_impl(builder, values);
    return std::make_unique<OwnedTable>(std::move(keys_tbl));
}

std::unique_ptr<OwnedTable> groupby_execute_values(
    GroupByBuilder& builder,
    const OwnedTable& values)
{
    auto [keys_tbl, value_cols] = execute_impl(builder, values);
    auto result = std::make_unique<cudf::table>(std::move(value_cols));
    return std::make_unique<OwnedTable>(std::move(result));
}

// ── GroupBy Scan ──────────────────────────────────────────────

namespace {

std::unique_ptr<cudf::groupby_scan_aggregation> make_scan_agg(int32_t agg_kind) {
    switch (agg_kind) {
        case 0: return cudf::make_sum_aggregation<cudf::groupby_scan_aggregation>();
        case 2: return cudf::make_min_aggregation<cudf::groupby_scan_aggregation>();
        case 3: return cudf::make_max_aggregation<cudf::groupby_scan_aggregation>();
        case 11: return cudf::make_count_aggregation<cudf::groupby_scan_aggregation>();
        case 12: return cudf::make_rank_aggregation<cudf::groupby_scan_aggregation>(
            cudf::rank_method::FIRST, cudf::order::ASCENDING,
            cudf::null_policy::INCLUDE, cudf::null_order::AFTER,
            cudf::rank_percentage::NONE);
        default:
            throw std::runtime_error("unknown groupby scan aggregation kind: " + std::to_string(agg_kind));
    }
}

} // anonymous namespace

std::unique_ptr<GroupByScanBuilder> groupby_scan_new(const OwnedTable& keys) {
    return std::make_unique<GroupByScanBuilder>(keys.view());
}

void groupby_scan_add_request(
    GroupByScanBuilder& builder,
    int32_t col_idx,
    int32_t agg_kind)
{
    builder.requests.emplace_back(
        static_cast<cudf::size_type>(col_idx),
        make_scan_agg(agg_kind));
}

std::unique_ptr<OwnedTable> groupby_scan_execute(
    GroupByScanBuilder& builder,
    const OwnedTable& values)
{
    cudf::groupby::groupby gb(builder.keys_view);

    // Group requests by column index.
    std::map<cudf::size_type, std::vector<std::unique_ptr<cudf::groupby_scan_aggregation>>> grouped;
    for (auto& [col_idx, agg] : builder.requests) {
        grouped[col_idx].push_back(std::move(agg));
    }

    std::vector<cudf::groupby::scan_request> requests;
    requests.reserve(grouped.size());
    for (auto& [col_idx, aggs] : grouped) {
        if (col_idx < 0 || col_idx >= values.view().num_columns()) {
            throw std::out_of_range(
                "scan column index " + std::to_string(col_idx) +
                " out of range [0, " + std::to_string(values.view().num_columns()) + ")");
        }
        cudf::groupby::scan_request req;
        req.values = values.view().column(col_idx);
        req.aggregations = std::move(aggs);
        requests.push_back(std::move(req));
    }

    auto [keys_result, scan_results] = gb.scan(requests);

    // Flatten: keys + all scan result columns.
    auto key_cols = keys_result->release();
    std::vector<std::unique_ptr<cudf::column>> all_cols;
    all_cols.reserve(key_cols.size() + scan_results.size());
    for (auto& c : key_cols) all_cols.push_back(std::move(c));
    for (auto& result_set : scan_results) {
        for (auto& col : result_set.results) {
            all_cols.push_back(std::move(col));
        }
    }

    auto combined = std::make_unique<cudf::table>(std::move(all_cols));
    return std::make_unique<OwnedTable>(std::move(combined));
}

// ── GroupBy Get Groups ────────────────────────────────────────

std::unique_ptr<GroupByGroupsResult> groupby_get_groups(
    const OwnedTable& keys)
{
    cudf::groupby::groupby gb(keys.view());
    auto groups = gb.get_groups();

    auto result = std::make_unique<GroupByGroupsResult>();
    result->keys = std::make_unique<OwnedTable>(std::move(groups.keys));

    // Convert offsets vector to a column.
    auto offsets_col = std::make_unique<cudf::column>(
        cudf::data_type{cudf::type_id::INT32},
        static_cast<cudf::size_type>(groups.offsets.size()),
        rmm::device_buffer(groups.offsets.data(),
                           groups.offsets.size() * sizeof(cudf::size_type),
                           cudf::get_default_stream()),
        rmm::device_buffer{}, 0);
    result->offsets = std::make_unique<OwnedColumn>(std::move(offsets_col));

    result->values = nullptr;  // No values requested
    return result;
}

std::unique_ptr<GroupByGroupsResult> groupby_get_groups_with_values(
    const OwnedTable& keys,
    const OwnedTable& values)
{
    cudf::groupby::groupby gb(keys.view());
    auto groups = gb.get_groups(values.view());

    auto result = std::make_unique<GroupByGroupsResult>();
    result->keys = std::make_unique<OwnedTable>(std::move(groups.keys));

    // Convert offsets vector to a column.
    auto offsets_col = std::make_unique<cudf::column>(
        cudf::data_type{cudf::type_id::INT32},
        static_cast<cudf::size_type>(groups.offsets.size()),
        rmm::device_buffer(groups.offsets.data(),
                           groups.offsets.size() * sizeof(cudf::size_type),
                           cudf::get_default_stream()),
        rmm::device_buffer{}, 0);
    result->offsets = std::make_unique<OwnedColumn>(std::move(offsets_col));

    result->values = std::make_unique<OwnedTable>(std::move(groups.values));
    return result;
}

std::unique_ptr<OwnedTable> groupby_groups_take_keys(GroupByGroupsResult& result) {
    return std::move(result.keys);
}

std::unique_ptr<OwnedColumn> groupby_groups_take_offsets(GroupByGroupsResult& result) {
    return std::move(result.offsets);
}

std::unique_ptr<OwnedTable> groupby_groups_take_values(GroupByGroupsResult& result) {
    if (!result.values) {
        throw std::runtime_error("no values in groups result");
    }
    return std::move(result.values);
}

// ── GroupBy Replace Nulls ─────────────────────────────────────

std::unique_ptr<OwnedTable> groupby_replace_nulls(
    const OwnedTable& keys,
    const OwnedTable& values,
    rust::Slice<const int32_t> policies)
{
    cudf::groupby::groupby gb(keys.view());

    std::vector<cudf::replace_policy> replace_policies;
    replace_policies.reserve(policies.size());
    for (auto p : policies) {
        replace_policies.push_back(
            p == 0 ? cudf::replace_policy::PRECEDING : cudf::replace_policy::FOLLOWING);
    }

    auto [keys_result, values_result] = gb.replace_nulls(
        values.view(), replace_policies);

    // Combine keys + replaced values.
    auto key_cols = keys_result->release();
    auto val_cols = values_result->release();
    std::vector<std::unique_ptr<cudf::column>> all_cols;
    all_cols.reserve(key_cols.size() + val_cols.size());
    for (auto& c : key_cols) all_cols.push_back(std::move(c));
    for (auto& c : val_cols) all_cols.push_back(std::move(c));

    auto combined = std::make_unique<cudf::table>(std::move(all_cols));
    return std::make_unique<OwnedTable>(std::move(combined));
}

} // namespace cudf_shims
