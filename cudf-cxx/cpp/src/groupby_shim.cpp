#include "groupby_shim.h"
#include <cudf/groupby.hpp>
#include <cudf/table/table.hpp>
#include <cudf/column/column.hpp>
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

} // namespace cudf_shims
