#include "partitioning_shim.h"
#include <cudf/partitioning.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <vector>

namespace cudf_shims {

std::unique_ptr<PartitionResult> hash_partition(
    const OwnedTable& table,
    rust::Slice<const int32_t> columns_to_hash,
    int32_t num_partitions)
{
    std::vector<cudf::size_type> cols(columns_to_hash.begin(), columns_to_hash.end());
    auto [result, offsets] = cudf::hash_partition(
        table.view(),
        cols,
        num_partitions);
    auto pr = std::make_unique<PartitionResult>();
    pr->table = std::make_unique<OwnedTable>(std::move(result));
    std::vector<int32_t> offsets_i32(offsets.begin(), offsets.end());
    pr->offsets = std::move(offsets_i32);
    return pr;
}

std::unique_ptr<PartitionResult> round_robin_partition(
    const OwnedTable& table,
    int32_t num_partitions)
{
    auto [result, offsets] = cudf::round_robin_partition(
        table.view(),
        num_partitions);
    auto pr = std::make_unique<PartitionResult>();
    pr->table = std::make_unique<OwnedTable>(std::move(result));
    std::vector<int32_t> offsets_i32(offsets.begin(), offsets.end());
    pr->offsets = std::move(offsets_i32);
    return pr;
}

std::unique_ptr<PartitionResult> partition(
    const OwnedTable& table,
    const OwnedColumn& partition_map,
    int32_t num_partitions)
{
    auto [result, offsets] = cudf::partition(
        table.view(),
        partition_map.view(),
        num_partitions);

    auto pr = std::make_unique<PartitionResult>();
    pr->table = std::make_unique<OwnedTable>(std::move(result));
    std::vector<int32_t> offsets_i32(offsets.begin(), offsets.end());
    pr->offsets = std::move(offsets_i32);
    return pr;
}

std::unique_ptr<OwnedTable> partition_result_table(
    std::unique_ptr<PartitionResult> result)
{
    return std::move(result->table);
}

rust::Vec<int32_t> partition_result_offsets(
    const PartitionResult& result)
{
    rust::Vec<int32_t> out;
    out.reserve(result.offsets.size());
    for (auto v : result.offsets) {
        out.push_back(v);
    }
    return out;
}

} // namespace cudf_shims
