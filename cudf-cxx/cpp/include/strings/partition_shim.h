#pragma once

#include <cudf/strings/split/partition.hpp>
#include <memory>
#include "rust/cxx.h"
#include "column_shim.h"
#include "table_shim.h"

namespace cudf_shims {

std::unique_ptr<OwnedTable> str_partition(
    const OwnedColumn& col, rust::Str delimiter);
std::unique_ptr<OwnedTable> str_rpartition(
    const OwnedColumn& col, rust::Str delimiter);

} // namespace cudf_shims
