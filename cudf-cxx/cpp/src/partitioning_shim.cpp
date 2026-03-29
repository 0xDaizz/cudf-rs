#include "partitioning_shim.h"
#include <cudf/partitioning.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <vector>

namespace cudf_shims {

std::unique_ptr<OwnedTable> hash_partition(
    const OwnedTable& table,
    rust::Slice<const int32_t> columns_to_hash,
    int32_t num_partitions)
{
    std::vector<cudf::size_type> cols(columns_to_hash.begin(), columns_to_hash.end());
    auto [result, offsets] = cudf::hash_partition(
        table.view(),
        cols,
        num_partitions);
    return std::make_unique<OwnedTable>(std::move(result));
}

std::unique_ptr<OwnedTable> round_robin_partition(
    const OwnedTable& table,
    int32_t num_partitions)
{
    auto [result, offsets] = cudf::round_robin_partition(
        table.view(),
        num_partitions);
    return std::make_unique<OwnedTable>(std::move(result));
}

} // namespace cudf_shims
