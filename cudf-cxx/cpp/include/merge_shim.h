#pragma once

#include <cudf/merge.hpp>
#include <cudf/table/table.hpp>
#include <memory>
#include <vector>
#include "rust/cxx.h"
#include "table_shim.h"

namespace cudf_shims {

/// Merge two pre-sorted tables into a single sorted table.
std::unique_ptr<OwnedTable> merge_tables(
    const OwnedTable& left,
    const OwnedTable& right,
    rust::Slice<const int32_t> key_cols,
    rust::Slice<const int32_t> orders,
    rust::Slice<const int32_t> null_orders);

} // namespace cudf_shims
