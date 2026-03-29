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

} // namespace cudf_shims
