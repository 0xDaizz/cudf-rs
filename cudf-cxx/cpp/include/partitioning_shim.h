#pragma once

#include <cudf/partitioning.hpp>
#include <cudf/table/table.hpp>
#include <cudf/column/column.hpp>
#include <memory>
#include <vector>
#include "rust/cxx.h"
#include "column_shim.h"
#include "table_shim.h"

namespace cudf_shims {

/// Partition a table by hashing the specified columns.
std::unique_ptr<OwnedTable> hash_partition(
    const OwnedTable& table,
    rust::Slice<const int32_t> columns_to_hash,
    int32_t num_partitions);

/// Partition a table using round-robin assignment.
std::unique_ptr<OwnedTable> round_robin_partition(
    const OwnedTable& table,
    int32_t num_partitions);

/// Result of a partition operation: table + offsets.
struct PartitionResult {
    std::unique_ptr<OwnedTable> table;
    std::vector<int32_t> offsets;

    int32_t num_offsets() const {
        return static_cast<int32_t>(offsets.size());
    }

    int32_t get_offset(int32_t index) const {
        return offsets.at(index);
    }
};

/// Partition a table using a partition map column.
/// Returns a PartitionResult with the reordered table and offsets.
std::unique_ptr<PartitionResult> partition(
    const OwnedTable& table,
    const OwnedColumn& partition_map,
    int32_t num_partitions);

/// Extract the table from a PartitionResult.
std::unique_ptr<OwnedTable> partition_result_table(
    std::unique_ptr<PartitionResult> result);

/// Get the offsets from a PartitionResult as a vec.
rust::Vec<int32_t> partition_result_offsets(
    const PartitionResult& result);

} // namespace cudf_shims
